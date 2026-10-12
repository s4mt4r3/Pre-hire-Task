# Architecture — Story Discovery & Summarization

A user describes their mood in free text. The system finds real uplifting news stories that fit it, then shows each one with a category, a mood-fit score, a paraphrased version and a 70–90 word summary, plus the source name and URL. Everything runs in one Colab notebook on a free T4 and uses only open-weight Hugging Face models.

Items marked **UNVERIFIED** have not been checked against the live source: model IDs, licenses, VRAM figures, library APIs and feed URLs. They need checking in Colab before submission (see CLAUDE.md rule 2).

---

## 1. Flow

```
 ┌──────────────────────────────┐
 │ Gradio UI                    │  mood text, number of stories (N)
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 1. Mood/Intent agent         │  small LLM, JSON output
 │    mood text -> MoodIntent   │  fallback: keyword rules if JSON parse fails
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 2. Retrieval agent (tool)    │  no LLM
 │    RSS feeds -> entries      │  feedparser
 │    keyword prefilter -> top K│  cheap overlap score vs MoodIntent.keywords
 │    fetch + extract text      │  trafilatura; fall back to the feed's full text
 │    -> list[Article]          │  bundled fallback stories if < 5 usable
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 3. Classifier/Ranker agent   │  mid LLM, JSON output, batched (4 per call)
 │    category (fixed list)     │
 │    is_uplifting + reason     │  drop if not uplifting
 │    mood_fit_score 0-10       │  rank by fit + category bonus, top 5
 │    -> list[ClassifiedStory]  │
 └──────────────┬───────────────┘
                ▼
   recommend() returns here: top 5 + story_ids + trace + metrics (JSON)
   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
   get_story_details(story_id): runs only when the user opens a story;
   cached per story_id
                ▼
 ┌──────────────────────────────┐
 │ 4. Writer agent              │  large LLM, two calls per story
 │    (a) paraphrase            │  generated once
 │    (b) summary  ◄────────┐   │
 └──────────────┬───────────┼───┘
                ▼           │ retry with feedback
 ┌──────────────────────────┼───┐  ("you wrote 64 words, need 70-90")
 │ 5. Validator (Python)    │   │
 │    70 <= words <= 90 ? ──┘   │  up to MAX_SUMMARY_RETRIES
 │    pass, or retries used up  │  if retries run out: closest attempt, flagged
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ StoryDetail -> UI (JSON)     │  story view + trace + metrics, cached
 └──────────────────────────────┘
```

The pipeline runs **stage by stage across the whole batch**: all articles are classified, then all stories are written. It does not run each story through every stage before starting the next. This keeps each model's calls together, which matters if models are loaded sequentially (§3).

---

## 2. Agents

### 2.1 Mood/Intent agent (small LLM)
- **Input:** raw mood string, cut to `MAX_MOOD_CHARS`. An empty string skips the model and returns a general intent.
- **Output:** `MoodIntent`. The model returns JSON `{emotion, desired_tone, target_categories[], keywords[]}`; each field is validated:
  - 0–3 categories from the fixed list
  - 3–8 concrete headline keywords, with negative feeling words dropped
  - `parse_method`: `"json"`, `"regex"` or `"default"`
- **Prompt:** the named constants `MOOD_SYSTEM_PROMPT`, `MOOD_USER_TEMPLATE` and `MOOD_FEWSHOT`. The system prompt has the exact JSON shape with `<...>` placeholders and the category list inserted verbatim, followed by one worked example. Greedy, ≤160 new tokens.
- **Robustness:**
  - `parse_model_output` strips code fences, finds the first balanced `{...}`, then tries `json.loads`, then a repair pass and a Python literal. If all of that fails, it extracts individual fields by regex. It never raises.
  - Invalid or missing categories are guessed with `CATEGORY_HINTS`; missing keywords come from the mood's content words; emotion and tone get defaults.
  - A failed model call is logged, and the rule-based intent is used instead.

### 2.2 Retrieval agent (no LLM)
- **Input:** `MoodIntent`, plus an optional `preferred_source`.
- **Steps:**
  1. **Fetch feeds.** Read every feed in `RSS_SOURCES` in parallel: `requests` fetches the feed and `feedparser` parses the bytes.
     - Timeouts are (connect 5 s, read `HTTP_TIMEOUT_S`), with up to `HTTP_RETRIES` retries and backoff on connection errors, 429 and 5xx.
     - A failing feed is logged and skipped.
     - Parsed feeds, and the article fetches in step 4, are cached for `FEED_CACHE_TTL_S`. The startup cell `check_feeds()` prints which feeds responded.
  2. **Dedupe** by normalized URL (tracking parameters stripped), then by normalized title.
  3. **Rank** entries by keyword overlap (title ×2, RSS summary ×1) with newer stories first on ties.
     - The preferred source, if one was named, ranks first and the others fill the remaining slots.
     - The name is matched to a config source by substring or `difflib` close match. An unmatched name is logged, and all sources are used.
  4. **Fetch and extract** the top `2 × PREFILTER_K` URLs in parallel.
     - Fetching uses an honest project User-Agent and at most 2 concurrent requests per host.
     - A page that errors (HTTP ≥ 400, unreachable, non-HTML) counts as unresolved and is dropped. The final URL after redirects is kept.
     - trafilatura extracts the text, plus the publish date if the feed lacks one. Short extractions fall back to the feed's full content (`text_source="rss_content"`) when it has any. Text still under `MIN_ARTICLE_CHARS` is dropped.
     - Duplicates found after redirects are dropped, and text is truncated to `MAX_ARTICLE_CHARS`.
  5. Keep the first `PREFILTER_K` usable articles. This caps the number of LLM calls on the T4.
- **Output:** `RetrievalResult(articles, used_fallback, feed_statuses, preferred_source)`.
- **Fallback:** if fewer than `MIN_USABLE_STORIES` (5) live articles survive, add bundled stories (8 original placeholder stubs, not real news) up to `PREFILTER_K`.
  - They are labeled source "Built-in sample (offline fallback)", use `example.org` URLs and have `text_source="fallback"`.
  - The trace logs `FALLBACK USED`, and the UI must flag them.
- **Known source quirks** (found 2026-10-12):
  - Good News Network sits behind Cloudflare, which rejects the bot User-Agent on some pages (403). Those articles are dropped and logged; the code does not pretend to be a browser.
  - Sunny Skyz's feed writes `and#039;` for `&#039;`; `_strip_html` repairs it.

Candidate feeds. All five returned valid RSS from a local machine on 2026-10-12, and their articles extracted cleanly. **UNVERIFIED:** reachability from Colab's IP range.

| Source | Feed URL |
|---|---|
| Good News Network | `https://www.goodnewsnetwork.org/feed/` |
| Positive News | `https://www.positive.news/feed/` |
| Reasons to be Cheerful | `https://reasonstobecheerful.world/feed/` |
| The Optimist Daily | `https://www.optimistdaily.com/feed/` |
| Sunny Skyz | `https://www.sunnyskyz.com/rss_tebow.php` |

### 2.3 Classifier/Ranker agent (mid LLM)
- **Input:** `MoodIntent` + the articles. Each prompt holds the title, the source and the first `CLASSIFIER_EXCERPT_CHARS` (1,500) characters, cut at a word boundary: about 350 tokens, under 1k per prompt.
- **Prompt:** `CLASSIFIER_SYSTEM_PROMPT` + `CLASSIFIER_USER_TEMPLATE`. The system prompt explicitly defines what is not uplifting and anchors the score at 10/7/4/0. There is no few-shot example, since the prompt runs once per story. The prompt deliberately leaves out the reader's preferred categories. With them included, a test model copied a preferred category onto unrelated stories; the preference is applied in code as the ranking bonus instead.
- **Execution:** `llm.generate_batch` in groups of `CLASSIFIER_BATCH_SIZE` (4), using left-padded batched generation, greedy, ≤120 new tokens. Unparsable replies get one batched repair retry with `JSON_RETRY_PROMPT`.
- **Output (JSON):** `{category, is_uplifting, mood_fit_score (0-10), one_line_reason}` becomes a `ClassifiedStory`.
  - **Category:** off-list values are mapped by substring or close match. If that fails, they are guessed from the title, else set to `"Other Uplifting"`.
  - **Missing score:** 3.
  - **Missing `is_uplifting`:** true, since every source is a curated positive-news feed.
  - Every default used is noted in the trace.
- **Ranking:** `final_score = mood_fit_score + CATEGORY_MATCH_BONUS (2, if category ∈ target_categories) + 0.1 × min(prefilter_score, 5)`.
  - Drop stories where `is_uplifting` is false, sort by `final_score` (newest first on ties) and return the top `CLASSIFIER_TOP_N` (5), plus the rejected list.
  - Never pad with stories that failed the check. If fewer than N live stories pass, the orchestrator tops up with the labeled built-in stories (classified the same way), so the UI always gets at least 5. This is flagged as `used_fallback` with a note.

### 2.4 Writer agent (large LLM)
- **Input:** the top `ClassifiedStory`s (article text cut to `WRITER_SOURCE_CHARS` = 4,000 characters, about 1k tokens) + `MoodIntent.desired_tone`.
- **Two focused plain-text calls per story, not one structured call.** Long prose inside JSON is fragile for open models, and separate calls let a retry regenerate only the summary.
  - **(a) Paraphrase:** shorter, in its own words, 120–200 words. No copied sentences, names and numbers kept exact, nothing added, in the reader's tone.
  - **(b) Summary:** one paragraph of 70–90 words (aim for 80): who, what, and why it's good news.
  - Both share `WRITER_SYSTEM_PROMPT`, which forbids invented facts and asks for plain prose. The templates are named constants (`PARAPHRASE_USER_TEMPLATE`, `SUMMARY_USER_TEMPLATE`, `SUMMARY_FEEDBACK_TEMPLATE`), each starting with the mock's task marker.
- **Batched rounds across stories** (`WRITER_BATCH_SIZE` = 5): all paraphrases, then all first summaries, then up to 3 retry rounds containing only the misses. That's 2 batched calls when everything passes and at most 5.
- **Retries:** a retry prompt holds the base summary prompt, the model's latest attempt and the exact-count feedback. Only the latest attempt is included, so the prompt doesn't grow.
- **Sampling:** `temperature = 0.7`, so wording drifts from the source and retries differ. Limits are `max_new_tokens` = 360 for the paraphrase and 180 for the summary.
- **Failures:** a failed batch leaves its stories with empty text, which then goes through the Validator's last-resort path. Nothing raises.

### 2.5 Validator (plain Python)
- **Word count:** `count_words` counts whitespace-separated tokens containing at least one letter or digit, so stray punctuation and em dashes don't count. The UI shows the same count.
- **Cleaning before counting:** `clean_writer_output` strips "Here is…:" and "Summary:" lead-ins, markdown headings and bold markers, quotes wrapped around the whole reply, and trailing "(82 words)" notes.
- **Feedback** (`summary_feedback`): the exact count, the 70–90 range, and how many words to add or remove (toward 80), e.g. "your summary has 64 words, but it must be between 70 and 90 words. Add about 16 words…". The loop runs up to `MAX_SUMMARY_RETRIES` (3) times.
- **Last resort** (`finalize_summary`), applied to the attempt closest to the range:
  - **Too long:** trim at the last sentence boundary that leaves 70–90 words. If none works, hard-cut at 90 words and end with "…".
  - **Too short:** append unseen sentences from the paraphrase, then the source, until 70 or more. Overshooting 90 is handled by the same trim.
  - **The result is always flagged:** `summary_status` is `trimmed`, `extended` or `out_of_range`, and the reason goes in `validation_notes`.
- **Copy check** (`copy_check`) on the paraphrase against the full source, using lowercased word tokens:
  - **Longest common n-gram** (dynamic programming): 12 or more words flags it.
  - **5-gram overlap:** above 30% of the paraphrase's 5-grams found in the source flags it.
  - A flag sets `copy_flagged` and quotes the shared run in the notes. The summary is checked the same way, but only noted.

### 2.6 Orchestrator (`StoryOrchestrator`)
- **Two steps, so the slow Writer only runs for stories the user opens:**
  - `recommend(mood, n=5, preferred_source=None)` runs Mood → Retrieval → Classifier/Ranker and returns the top `n` recommendations, each with a `story_id`.
  - `get_story_details(story_id)` runs the Writer + Validator **lazily** for one story and caches the result by `story_id`, a SHA-1 of the URL, so repeat opens are cache hits. A cached story keeps the tone of the mood that first produced it.
- **AgentTrace:** an ordered list of `TraceStep(agent, model, status, duration_s, output_preview, events)`.
  - Status is `ok`, `fallback`, `error`, `cache_hit` or `skipped`.
  - The Validator's final checks are timed inside `write_stories` and split into their own step.
  - Retries that succeed are not fallbacks; they're counted under metrics.
- **Metrics,** per agent and in total:
  - calls, input/output tokens (real tokenizer, sliced from `llm.call_log`), wall time
  - tokens/sec: output tokens ÷ the agent's wall time, because batched rows share one generate(). Mock mode reports 0.
  - peak GPU memory, plus currently allocated memory in the total
  - retries: classifier JSON repairs and summary re-prompts
  - cache hits: feeds, articles and story details
- **Robustness:** each step runs under `_RunRecorder.run`. An unexpected exception becomes a safe default (rule-based intent, fallback stories, default verdicts) and an `error` step. Unknown story ids return an `error` field instead of raising.
- **Backend:** the orchestrator owns its `llm` backend and swaps it in for each call (`use_backend`), so a mock orchestrator also works on a GPU runtime.
- **JSON-safe:** `to_jsonable` converts every response to plain dicts and lists, with dates as ISO strings and NaN/inf as null.

---

## 3. Models

Requirements: at least 3 open-weight LLMs, no gating (so no HF token is needed for "Run All"), and a free T4 (about 15 GB usable VRAM, **fp16 only**: no bf16, no FlashAttention-2), with about 12 GB system RAM.

| Role | Model | Params | License | Precision | Approx VRAM (weights) |
|---|---|---|---|---|---|
| Mood/Intent (small) | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Apache-2.0 | fp16 | ~3.1 GB |
| Classifier/Ranker (mid) | `microsoft/Phi-3.5-mini-instruct` | 3.8B | MIT | 4-bit NF4 | ~2.4 GB |
| Writer (large) | `Qwen/Qwen2.5-7B-Instruct` | 7.6B | Apache-2.0 | 4-bit NF4 | ~5.5 GB |

**Verified 2026-10-12 via the HF API:** all three IDs (plus the OLMo swap option) exist, are ungated, and carry the licenses above. The parameter counts come from their safetensors metadata, and all checkpoints are stored in BF16.

**UNVERIFIED:** the VRAM figures (estimated from parameter count; measure with `torch.cuda.max_memory_allocated()`).

Why these:
- **Qwen2.5-1.5B** is reliable at short JSON output for its size, and the intent step is short and constrained.
- **Phi-3.5-mini** handles reasoning-style judgments ("is this genuinely uplifting?") well for its size, has a permissive MIT license, and comes from a different model family than the other two.
- **Qwen2.5-7B** gives the best prose and instruction-following (including word-count targets) among ungated models that fit in 4-bit.
- **Swap options** (also UNVERIFIED):
  - `allenai/OLMo-2-1124-7B-Instruct` (Apache-2.0) as the writer, if graders favor three distinct families.
  - `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Apache-2.0) as the small model.
  - Avoided on purpose: Llama 3.x, Gemma and Mistral checkpoints, which are gated or need license acceptance on HF.

### Loading strategy
- **Default: all three resident.** The weights total about 11 GB, plus about 1.5–2 GB for KV cache and activations at our prompt sizes (articles truncated, `max_new_tokens` ≤ 400). The estimate is about 13 GB, under the T4's ~15 GB. Each model loads once, and each request costs only generation time.
- **Fallback: `LOAD_STRATEGY = "sequential"`.** Load a stage's model, run the whole batch through that stage, then `del model`, `gc.collect()` and `torch.cuda.empty_cache()` before the next stage. Peak VRAM is about 6–7 GB, but every request pays the reload time: about 20–60 s for the 7B from local disk, **UNVERIFIED**.
- The notebook prints VRAM after loading. If loading in `"resident"` mode raises CUDA OOM, it frees memory and switches to `"sequential"` automatically, so "Run All" never needs an edit.
- Weights download once to the Colab disk cache: about 3 + 7.6 + 15 GB for the full-precision checkpoints, which are quantized at load time. **UNVERIFIED:** whether that fits in free Colab disk alongside everything else. If not, use pre-quantized repos for the 4-bit models.

---

## 4. Data schemas

These are the dataclasses passed between agents. Every agent also appends `TraceEvent`s, which the UI shows so the multi-agent flow is visible.

```python
CATEGORIES: list[str]           # §5, fixed; the only values a category field may take

@dataclass
class TraceEvent:
    agent: str                  # "mood" | "retrieval" | "classifier" | "writer" | "validator"
    message: str                # human-readable step description
    elapsed_s: float
    fallback_used: bool = False

@dataclass
class MoodIntent:
    raw_mood: str
    emotion: str                # e.g. "stressed"
    desired_tone: str           # e.g. "calm and reassuring"; passed to Classifier and Writer
    target_categories: list[str]  # 0–3, subset of CATEGORIES; empty = no preference
    keywords: list[str]         # 3–8 lowercase headline words used by Retrieval ranking
    parse_method: str           # "json" | "regex" | "default"

@dataclass
class Article:
    title: str
    url: str                    # canonical article URL (shown in UI)
    source_name: str            # e.g. "Good News Network"
    published: datetime | None
    text: str                   # extracted text, truncated to MAX_ARTICLE_CHARS
    text_source: str            # "extracted" | "rss_content" | "fallback"
    prefilter_score: float      # keyword overlap from Retrieval

@dataclass
class ClassifiedStory:
    article: Article
    category: str               # exactly one of CATEGORIES
    is_uplifting: bool
    mood_fit_score: int         # 0–10
    reason: str                 # model's one_line_reason
    category_match: bool        # category ∈ intent.target_categories
    final_score: float          # ranking key (§2.3)
    parse_method: str           # "json" | "regex" | "default"

@dataclass
class CallMetric:
    purpose: str                # "paraphrase", "summary attempt 2", ...
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float      # batched calls report the whole batch's time
    tokens_per_second: float
    peak_gpu_memory_gb: float | None
    backend: str                # "hf" | "mock"

@dataclass
class StoryDetail:
    title: str
    source: str
    url: str
    published: datetime | None
    category: str
    mood_fit_score: int
    final_score: float
    paraphrase: str
    paraphrase_word_count: int
    summary: str
    word_count: int             # summary words (count_words)
    summary_passed: bool        # 70–90
    summary_status: str         # "ok" | "ok_after_retry" | "trimmed" | "extended" | "out_of_range"
    summary_attempt_word_counts: list[int]
    copy_longest_ngram: int
    copy_5gram_ratio: float
    copy_flagged: bool
    is_fallback_story: bool     # built-in placeholder, not real news
    validation_notes: list[str]
    calls: list[CallMetric]

@dataclass
class TraceStep:
    agent: str                  # "mood" | "retrieval" | "classifier" | "writer" | "validator"
    model: str                  # model id, "mock of <id>", or "none (...)" for Python agents
    status: str                 # "ok" | "fallback" | "error" | "cache_hit" | "skipped"
    duration_s: float
    output_preview: str         # ≤160 chars
    events: list[TraceEvent]

# Orchestrator responses (JSON via to_jsonable):
# recommend()          -> {run_id, mood, mock_llm, intent, recommendations[{story_id, rank, title, source, url,
#                          published, category, mood_fit_score, final_score, category_match, reason,
#                          is_fallback_story, details_cached}], rejected[], used_fallback, preferred_source,
#                          feeds[], notes[], trace[TraceStep], metrics{per_agent{...}, total{...}}}
# get_story_details()  -> {story_id, cache_hit, story: StoryDetail + story_id | None, error?, trace, metrics}
```

---

## 5. Categories

The assignment's 10 suggested categories, plus two additions it allows ("You may define additional categories"):

| # | Category | Origin |
|---|---|---|
| 1 | Success Through Hardship | assignment |
| 2 | Friendship | assignment |
| 3 | Kindness | assignment |
| 4 | Innovation | assignment |
| 5 | Leadership | assignment |
| 6 | Community Impact | assignment |
| 7 | Personal Growth | assignment |
| 8 | Perseverance | assignment |
| 9 | Education | assignment |
| 10 | Health & Well-being | assignment |
| 11 | **Animals & Nature** | extra: the feeds carry many wildlife/nature stories that fit none of the 10 |
| 12 | **Other Uplifting** | extra: catch-all, so an off-list model answer never forces a wrong category |

---

## 6. LLM interface and `USE_MOCK_LLM`

All three LLM agents call one method on a global `llm` object, so the real and mock backends can be swapped in a single place:

```
llm.generate(model_key, messages, max_new_tokens, temperature) -> GenerationResult
    model_key ∈ {"mood", "classifier", "writer"}
    GenerationResult: text, input/output token counts (real tokenizer), elapsed_seconds,
                      tokens_per_second, peak_gpu_memory_gb, backend ("hf" | "mock")
```

`llm` is a `ModelManager` (real) or a `MockModelManager`.

**Contract with the Writer agent:** the first line of each Writer prompt carries `TASK: PARAPHRASE` or `TASK: SUMMARY`, and a retry prompt includes `Validator feedback:`. The mock uses these markers to choose its 150-, 60- or 80-word reply.

- **`USE_MOCK_LLM = False`** (default on GPU): loads the HF models from §3 and runs `apply_chat_template` + `generate`.
- **`USE_MOCK_LLM = True`**: no `torch`/`transformers` model loading at all. It returns deterministic fake outputs per role, built from the input (e.g. the mood agent echoes keywords from the mood text; the classifier derives a category and score from the article title). It runs on CPU in seconds.
- **Default value:** `USE_MOCK_LLM = not torch.cuda.is_available()`, so "Run All" on a GPU runtime uses the real models with no edit. An explicit override can be set in the config cell.
- **The mock writer exercises the retry path on purpose.** Its first summary attempt is 60 words, so the Validator rejects it. The second attempt is 80 words. This shows in local testing that the feedback loop and the UI warning/attempt display work.
- Mock mode still uses real Retrieval, or the bundled fallback stories when live retrieval falls short, so feed parsing and extraction get tested too.
- The UI shows a visible "MOCK LLM" banner when the flag is on.

---

## 7. Gradio UI (`gr.Blocks`, Gradio 6.28)

- **Header:** project name, tagline, and a mode pill (real models + GPU name, or mock mode).
- **Inputs:**
  - mood textbox; Enter also searches
  - example-mood chips, filled client-side via `js=` so a fast click can't search with an empty mood
  - source dropdown ("Any" + the configured feeds)
  - a prominent **Tell Me a Story** button
- **Results** (left):
  - banners for mock mode, offline fallback, preferred source and notes
  - a compact story picker (`gr.Radio`, numbered like the cards)
  - 5 cards, each with rank, title, source/date, category badge, a mood-fit bar, a "matches your mood" tag and the classifier's reason
- **Story detail** (right): it loads lazily through `get_story_details()`.
  - category badge and copy-check tag, title, source and date, and a link (http(s) only)
  - summary with a word-count pill: green if 70–90, amber if trimmed or extended, red if out of range
  - paraphrase, and the validation notes in a `<details>`
- **Agent Trace accordion**, updated live:
  - Handlers are generators. The orchestrator runs in a worker thread with an `on_step` callback feeding a queue, and each finished step re-renders the timeline.
  - Rows are ✅ ok / ⚠️ fallback / ❌ error / ⚡ cache hit / ⏭️ skipped / ⏳ running / ○ pending, and each expands to its events.
- **Metrics accordion:** tiles for input/output tokens, time, tokens/sec, GPU memory (peak and allocated), retries and cache hits, plus a per-agent table. It combines the search metrics and the opened story's metrics.
- **Loading and errors:**
  - The button is disabled and relabelled while a request runs, and the detail panel shows a loading message.
  - An empty mood gets a friendly prompt.
  - Exceptions become friendly banners.
- **Safety:** `gr.HTML` doesn't sanitize, so every dynamic string is `html.escape`d.
- **Theme:** `gr.themes.Soft` (orange/rose/slate, Inter). Custom CSS uses Gradio theme variables, so light and dark both work.
- **Launch:** in Gradio 6, `theme` and `css` are passed to `launch()`, not `Blocks()`. The final cell calls `demo.queue(default_concurrency_limit=1).launch(share=True, ...)`; one request at a time because there's a single GPU and a global backend swap. Re-running the cell closes the previous app first.
  - **UNVERIFIED:** `share=True` tunnel behavior on Colab. It was not exercised locally, to avoid opening a public link.
- **Tested locally** (mock mode, headless Chromium):
  - the full flow in light and dark mode
  - the 390 px mobile layout: no horizontal scroll
  - empty mood, then a search
  - two browser tabs
  - a cache-hit reopen

## 8. Configuration (single config cell)

| Name | Default | Purpose |
|---|---|---|
| `USE_MOCK_LLM` | `not cuda available` | §6 |
| `LOAD_STRATEGY` | `"resident"` | `"resident"` or `"sequential"`; switches to sequential on OOM |
| `MODEL_IDS` | §3 table | one place to swap models |
| `RSS_SOURCES` | §2.2 table | |
| `PREFILTER_K` | 8 | articles sent to the Classifier |
| `MAX_ARTICLE_CHARS` | 6000 | prompt and VRAM bound |
| `MIN_ARTICLE_CHARS` | 400 | below this, use the RSS summary |
| `MAX_SUMMARY_RETRIES` | 3 | Validator loop |
| `SUMMARY_MIN_WORDS` / `SUMMARY_MAX_WORDS` | 70 / 90 | spec requirement |
| `HTTP_TIMEOUT_S` | 10 | per feed and per article |
| `RUN_EVAL` | `True` | run the 10-mood evaluation during Run All |
| `EVAL_STORIES_PER_MOOD` | 2 | stories per mood sent through Writer + Validator in the evaluation |

No API keys exist anywhere in the config. All models are ungated, so no HF token is needed. If one is ever added, it is read from Colab Secrets or an env var (CLAUDE.md rule 3).

---

## 9. Notebook layout (`notebook.py` → `.ipynb`)

Each code section gets a markdown cell before it explaining what it does and why.

Title, overview with an ASCII architecture diagram, how to run, contents; then **Manual steps** (required: none); then:

1. Install dependencies (pinned; **UNVERIFIED:** compatible with Colab's preinstalled torch without a restart)
2. Configuration (§8)
3. Environment check (GPU, VRAM, library versions)
4. Models and `ModelManager` / `MockModelManager` (§3, §6)
5. Model smoke test
6. Data schemas (§4)
7. Bundled fallback stories
8. Agent 2: Retrieval + feed health check + test
9. Robust JSON parsing
10. Prompt design + prompt constants (Mood, Classifier)
11. Agent 1: Mood/Intent
12. Agent 3: Classifier/Ranker + test (3 moods)
13. Agent 5: Validator (word count, cleaning, trim/extend, copy check)
14. Agent 4: Writer + test (Validator checks + end to end)
15. Orchestrator: `StoryOrchestrator.recommend()` / `get_story_details()` + end-to-end tests (mock; real if CUDA)
16. Evaluation (`RUN_EVAL`, default on): 10 moods → per-mood and per-stage tables + summary; Results & Analysis template; Known Limitations
17. Gradio UI: rendering helpers
18. Gradio UI: the app
19. Launch (`share=True`) + instructions: the final code cell
20. References: at the end of the notebook, as the assignment requires
21. AI Assistance Declaration (tool, major prompts, how output was reviewed and tested)

---

## 10. Build order

**Minimum requirements first:** schemas, config, mock backend, all five agents, orchestrator and a basic UI. Get this working end to end in mock mode on CPU, then on a real T4.

Only after that, consider extras: per-category feeds, result caching, a streaming UI, or a faithfulness check on the writer output.

## 11. Open questions (resolved 2026-10-12 against the assignment description)

1. **Categories:** the assignment's 10 suggested categories + "Animals & Nature" + "Other Uplifting" (§5).
2. **Paraphrase length / story count:** the assignment says only "a shorter paraphrased version" and "at least 5 stories". 120–200 words and top 5 (topped up with labeled built-in stories if needed) satisfy both.
3. **Model families:** the assignment requires at least 3 open-weight LLMs, not distinct families; the three models used are distinct models.
