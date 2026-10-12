# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: T4
#     provenance: []
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     name: python
# ---

# %% [markdown]
# # Story Discovery & Summarization: a Multi-Agent System
#
# Describe how you feel in a sentence. A team of agents finds real, uplifting
# news stories that fit your mood:
# - It recommends the **top 5** stories, each with a category, a mood-fit
#   score, the source name and a link to the original article.
# - When you open one, it writes a **paraphrased version** in its own words
#   and a **70–90 word summary**, both checked by a validator.
#
# All models are open-weight and ungated, loaded from Hugging Face. **No
# proprietary APIs and no API keys are used.**
#
# ## Architecture
#
# ```text
#                   ┌──────────────────────────────────────────────────┐
#    your mood ───► │ Gradio UI: mood text, optional source, picks     │
#                   └───────────┬──────────────────────────┬───────────┘
#        recommend(mood)        │                          │ get_story_details(story_id)
#                               ▼                          ▼ (only for the story you open)
#  ┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
#  │ 1. Mood/Intent      Qwen2.5-1.5B     │  │ 4. Writer        Qwen2.5-7B (4-bit)  │
#  │    mood text → JSON intent           │  │    (a) paraphrase (b) summary        │
#  │             │                        │  │                 ▲  retry with the    │
#  │             ▼                        │  │                 │  exact word count  │
#  │ 2. Retrieval        Python tool      │  │ 5. Validator     Python              │
#  │    5 RSS feeds → trafilatura text    │  │    70–90 words? trim/extend if not;  │
#  │    → up to 8 articles (+ fallback)   │  │    copy + fact checks vs. the source │
#  │             │                        │  │                 │                    │
#  │             ▼                        │  │                 ▼                    │
#  │ 3. Classifier/Ranker  Phi-3.5-mini   │  │    StoryDetail (cached per story)    │
#  │    category · uplifting? · fit 0–10  │  └──────────────────────────────────────┘
#  │    → top 5 recommendations           │
#  └──────────────────────────────────────┘
#   Every step is recorded: Agent Trace (live in the UI) + metrics
#   (tokens in/out, time, tokens/s, GPU memory, retries, cache hits).
# ```
#
# | # | Agent | Kind | Job |
# |---|---|---|---|
# | 1 | Mood/Intent | small LLM (Qwen2.5-1.5B-Instruct, fp16) | mood text → emotion, desired tone, target categories, search keywords (JSON) |
# | 2 | Retrieval | Python tool, no LLM | fetches RSS feeds of uplifting news, extracts article text, keeps title, source name, URL and date |
# | 3 | Classifier/Ranker | mid LLM (Phi-3.5-mini-instruct, 4-bit) | assigns one category, checks the story is genuinely uplifting, scores mood fit, ranks |
# | 4 | Writer | large LLM (Qwen2.5-7B-Instruct, 4-bit) | paraphrased version + 70–90 word summary |
# | 5 | Validator | plain Python | enforces 70–90 words (feedback + retries, then trim/extend), copy check, fact check (names and numbers vs. the source) |
#
# ## How to run
#
# 1. In Colab: **Runtime → Change runtime type → T4 GPU → Save**.
# 2. **Runtime → Run all**.
# 3. Wait for the last code cell (section 19) to print a Gradio link
#    (`https://….gradio.live`). Open it, or use the interface shown inline
#    below that cell.
#
# The first run downloads about 25 GB of model weights, which takes several
# minutes. No edits are needed (see **Manual steps** below). Without a GPU,
# the notebook still runs end to end with fake model outputs
# (`USE_MOCK_LLM`), so the pipeline and UI can be tested on CPU.
#
# ## Contents
#
# 1. Install dependencies
# 2. Configuration
# 3. Environment check
# 4. Models and the ModelManager (model choice, mock backend)
# 5. Model smoke test
# 6. Data schemas
# 7. Bundled fallback stories
# 8. Agent 2: Retrieval (+ feed health check, test)
# 9. Robust JSON parsing
# 10. Prompt design (+ prompt templates)
# 11. Agent 1: Mood/Intent
# 12. Agent 3: Classifier/Ranker (+ test with 3 moods)
# 13. Agent 5: Validator
# 14. Agent 4: Writer (+ test)
# 15. Orchestrator (+ end-to-end tests: mock, and real if a GPU is present)
# 16. Evaluation (`RUN_EVAL`), Results & Analysis, Known Limitations
# 17. Gradio UI: rendering helpers
# 18. Gradio UI: the app
# 19. Launch the app (the final code cell)
# 20. References (at the end, as the assignment asks)
# 21. AI Assistance Declaration

# %% [markdown]
# ## Manual steps
#
# **Required: none.** "Runtime → Run all" works with no edits, no API keys,
# no file uploads and no Google Drive access.
#
# **Recommended before Run all:**
# - Select a **T4 GPU** (Runtime → Change runtime type). The notebook's
#   metadata asks Colab for a T4 (**UNVERIFIED:** that Colab honors this when
#   opening the file). Without a GPU, nothing fails: the notebook switches to
#   mock mode and says so.
#
# **Optional settings** in the Configuration cell (section 2). None are
# needed for a normal run:
# - `RUN_EVAL = False` skips the 10-mood evaluation. With real models it
#   runs about 20 Writer passes, which can take tens of minutes on a T4
#   (**UNVERIFIED:** exact time). The UI launches only after it, in the last
#   cell, so set this to `False` if you just want to try the app quickly.
#   `EVAL_STORIES_PER_MOOD` changes its size.
# - `LOAD_STRATEGY = "sequential"` forces one model in GPU memory at a time.
#   The notebook already switches to this automatically if it runs out of
#   memory.
# - The `USE_MOCK_LLM=1` or `0` environment variable forces mock or real
#   mode.
# - If you swap in a **gated** model, set `HF_TOKEN_REQUIRED = True` and add
#   an `HF_TOKEN` secret in Colab (key icon in the left sidebar). The default
#   models don't need this.
#
# **If something goes wrong:**
# - If Colab asks to restart the runtime after the install cell (not expected
#   with these pins), restart, then Run all again.
# - If the feeds are unreachable, the app uses the built-in placeholder
#   stories and labels them. Nothing needs editing.
#
# **For the author only, not for running:** fill in the `[FILL]` placeholders
# in "Results & Analysis" (section 16) from a T4 run.

# %% [markdown]
# ## 1. Install dependencies
#
# Installs the libraries the pipeline needs. `torch` is not installed here: the
# notebook uses the version Colab preinstalls, which matches Colab's CUDA driver.
#
# | Package | Used for |
# |---|---|
# | `transformers`, `accelerate` | loading and running the three LLMs |
# | `bitsandbytes` | 4-bit quantization so the mid and large models fit on a T4 |
# | `gradio` | the web UI |
# | `feedparser` | reading RSS feeds |
# | `trafilatura` | extracting the main article text from a web page |
# | `requests` | HTTP fetching with timeouts (already in Colab, so not reinstalled) |
#
# Versions are pinned for reproducibility. Each pin had been on PyPI for at
# least two weeks as of 2026-10-12. A pip dry run confirmed they install
# together on Colab's platform (Linux, Python 3.12). `requests` is not
# reinstalled: Colab already ships it. `%%capture` hides pip's long output.
# The GPU check cell further down prints the installed versions, so a failed
# install is still visible.

# %%
# %%capture
# Checked: these pins resolve together for Linux x86-64 / Python 3.12 (pip dry run) and need only torch>=2.5.
# requests is not pinned: Colab already ships it and its own tooling depends on it.
# UNVERIFIED: that installing over Colab's preinstalled packages never asks for a runtime restart.
# %pip install -q transformers==5.17.0 accelerate==1.15.0 bitsandbytes==0.50.2 gradio==6.28.0 feedparser==6.0.14 trafilatura==2.2.0

# %% [markdown]
# ## 2. Configuration
#
# This is the only place to change settings. Nothing here needs editing for a normal run.
#
# - **`MODEL_IDS` / `QUANTIZE_4BIT`**: the three open-weight models and which
#   of them load in 4-bit. All three are ungated on Hugging Face (checked
#   2026-10-12), so no access token is needed.
# - **`LOAD_STRATEGY`**: `"resident"` keeps all three models on the GPU, which
#   is faster per request. `"sequential"` loads one model at a time, which uses
#   less VRAM. The loader switches to sequential automatically if resident
#   loading runs out of memory.
# - **`CATEGORIES`**: the fixed list the Classifier must choose from. It
#   holds the assignment's 10 suggested categories (Success Through Hardship,
#   Friendship, Kindness, Innovation, Leadership, Community Impact, Personal
#   Growth, Perseverance, Education, Health & Well-being), plus two
#   additions: "Animals & Nature", because the feeds carry many wildlife
#   stories, and "Other Uplifting" as a catch-all.
# - **`SUMMARY_MIN_WORDS` / `SUMMARY_MAX_WORDS`**: the 70–90 word rule the
#   Validator enforces, with `MAX_SUMMARY_RETRIES` attempts.
# - **`RSS_SOURCES`**: the uplifting-news feeds the Retrieval agent reads.
# - **`USE_MOCK_LLM`**: detected automatically. It is `True` when there is no
#   CUDA GPU, in which case fake model outputs stand in for the LLMs. Setting
#   the environment variable `USE_MOCK_LLM=1` or `0` overrides it.
# - **`RUN_EVAL`**: runs the 10-mood evaluation (section 16) during "Run
#   all". With real models this can take tens of minutes, and the UI only
#   launches after it; set it to `False` to skip.
# - **Hugging Face token**: only read (from Colab Secrets) if
#   `HF_TOKEN_REQUIRED` is set to `True` after swapping in a gated model.
#   The default is `False`.

# %%
import os

MODEL_IDS = {
    "mood": "Qwen/Qwen2.5-1.5B-Instruct",  # Apache-2.0, 1.5B params
    "classifier": "microsoft/Phi-3.5-mini-instruct",  # MIT, 3.8B params
    "writer": "Qwen/Qwen2.5-7B-Instruct",  # Apache-2.0, 7.6B params
}
# T4 has 15 GB of VRAM; 4-bit on the two bigger models is what lets all three stay resident together.
# UNVERIFIED: VRAM estimates (~3.1 GB fp16 / ~2.4 GB 4-bit / ~5.5 GB 4-bit); measure in Colab.
QUANTIZE_4BIT = {"mood": False, "classifier": True, "writer": True}
LOAD_STRATEGY = "resident"
# T4 (compute capability 7.5) has no bf16 support, so the bf16 checkpoints are run in fp16.
# Checked (CPU, fp16 forward pass): peak activations stay far below fp16's 65504 limit, with no inf/NaN —
# Qwen2.5-1.5B ~6.6k, Phi-3.5-mini ~3.9k. UNVERIFIED: the same for Qwen2.5-7B (too large to test locally).
TORCH_DTYPE = "float16"

HF_TOKEN_REQUIRED = False

# The 10 categories suggested by the assignment, plus two additions it allows ("You may define additional
# categories"): the feeds carry many wildlife/nature stories that fit none of the 10, and a catch-all keeps an
# off-list model answer from forcing a wrong label.
CATEGORIES = [
    "Success Through Hardship",
    "Friendship",
    "Kindness",
    "Innovation",
    "Leadership",
    "Community Impact",
    "Personal Growth",
    "Perseverance",
    "Education",
    "Health & Well-being",
    "Animals & Nature",  # additional
    "Other Uplifting",  # additional catch-all
]

SUMMARY_MIN_WORDS = 70
SUMMARY_MAX_WORDS = 90
MAX_SUMMARY_RETRIES = 3

# Each feed returned valid RSS, and its articles extracted cleanly, from a local machine on 2026-10-12.
# The feed health check in the Retrieval section reports which ones respond at run time.
RSS_SOURCES = {
    "Good News Network": "https://www.goodnewsnetwork.org/feed/",  # UNVERIFIED: reachable from Colab IPs
    "Positive News": "https://www.positive.news/feed/",  # UNVERIFIED: reachable from Colab IPs
    "Reasons to be Cheerful": "https://reasonstobecheerful.world/feed/",  # UNVERIFIED: reachable from Colab IPs
    "The Optimist Daily": "https://www.optimistdaily.com/feed/",  # UNVERIFIED: reachable from Colab IPs
    "Sunny Skyz": "https://www.sunnyskyz.com/rss_tebow.php",  # UNVERIFIED: reachable from Colab IPs
}
PREFILTER_K = 8  # articles handed to the Classifier
MIN_USABLE_STORIES = 5  # below this many live articles, the bundled fallback stories are added
MAX_ARTICLE_CHARS = 6000
MIN_ARTICLE_CHARS = 400
HTTP_TIMEOUT_S = 10
HTTP_RETRIES = 2
FEED_CACHE_TTL_S = 15 * 60  # repeat UI requests within 15 min reuse feeds and articles instead of refetching
HTTP_USER_AGENT = "StoryDiscoveryNotebook/1.0 (educational demo; +https://colab.research.google.com)"

MAX_MOOD_CHARS = 500  # longer mood text is cut before it reaches the Mood model
CLASSIFIER_EXCERPT_CHARS = 1500  # ~350 tokens of each story is enough to judge category, tone and fit
CLASSIFIER_BATCH_SIZE = 4  # stories per batched generate() call; bounds KV-cache memory on the T4
CLASSIFIER_TOP_N = 5  # ranked stories handed to the Writer
CATEGORY_MATCH_BONUS = 2.0  # added to mood_fit_score when the story's category is one the reader asked for

WRITER_SOURCE_CHARS = 4000  # ~1k tokens of article per Writer prompt: enough for the main facts, fast on a T4
WRITER_BATCH_SIZE = 5  # stories per batched Writer call (= CLASSIFIER_TOP_N, so one batch per round)
WRITER_TEMPERATURE = 0.7  # sampling so paraphrases reword freely and summary retries actually differ
PARAPHRASE_MIN_WORDS = 120
PARAPHRASE_MAX_WORDS = 200
SUMMARY_TARGET_WORDS = 80  # aimed for in prompts; the middle of the 70–90 range leaves slack both ways
COPY_MAX_COMMON_NGRAM = 12  # a run of this many identical words shared with the source = copied phrasing
COPY_MAX_5GRAM_RATIO = 0.30  # or: more than 30% of the paraphrase's 5-word sequences appear in the source
FACT_MAX_UNSUPPORTED_NAMES = 2  # fact check warns on any number missing from the source, or on more names than this

RUN_EVAL = True  # set False to skip the evaluation (can take tens of minutes on a T4; the UI launches after it)
EVAL_STORIES_PER_MOOD = 2  # stories per mood run through the Writer + Validator; all 5 would multiply Writer time

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

_mock_override = os.environ.get("USE_MOCK_LLM")
USE_MOCK_LLM = (not HAS_CUDA) if _mock_override is None else _mock_override == "1"

HF_TOKEN = None
if HF_TOKEN_REQUIRED:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN is None:
        try:
            from google.colab import userdata

            HF_TOKEN = userdata.get("HF_TOKEN")
        except Exception as e:
            # userdata raises different exception types for "not in Colab", "secret missing" and
            # "notebook access not granted"; all three need the same fix from the user.
            raise RuntimeError(
                "A gated model is configured but no HF_TOKEN was found. In Colab, open the key icon "
                "in the left sidebar, add a secret named HF_TOKEN, and enable notebook access."
            ) from e
    print("Hugging Face token loaded from Colab Secrets / environment.")
else:
    print("No Hugging Face token needed: all configured models are ungated.")

print(f"USE_MOCK_LLM = {USE_MOCK_LLM}  (CUDA available: {HAS_CUDA})")

# %% [markdown]
# ## 3. Environment check
#
# Prints the GPU name, total and free VRAM, and the installed library
# versions. If the install cell above failed silently, it shows up here.
#
# - **No GPU:** a warning explains how to switch to a T4. The notebook keeps
#   going in mock mode rather than crashing, so "Run all" still finishes.
# - **GPU under ~14 GB:** a warning, because the resident loading strategy may
#   not fit.

# %%
from importlib.metadata import PackageNotFoundError, version

for pkg in ["torch", "transformers", "accelerate", "bitsandbytes", "gradio", "feedparser", "trafilatura", "requests"]:
    try:
        print(f"{pkg:<13} {version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg:<13} NOT INSTALLED")

if HAS_CUDA:
    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"\nGPU: {props.name} (compute capability {props.major}.{props.minor})")
    print(f"VRAM: {total_b / 1024**3:.1f} GB total, {free_b / 1024**3:.1f} GB free")
    if total_b / 1024**3 < 14:
        print("WARNING: less VRAM than a T4; resident loading may not fit and will fall back to sequential.")
else:
    print("\nWARNING: no CUDA GPU detected.")
    print("  To use the real models: Runtime → Change runtime type → T4 GPU, then Runtime → Run all.")
    print("  Continuing with USE_MOCK_LLM = True (fake model outputs) so the pipeline and UI still run.")

# %% [markdown]
# ## 4. Models and the ModelManager
#
# ### Which model does which job, and why
#
# | Agent | Model | License | Size | Precision on T4 | Approx. VRAM |
# |---|---|---|---|---|---|
# | Mood/Intent | `Qwen/Qwen2.5-1.5B-Instruct` | Apache-2.0 | 1.5B | fp16 | ~3.1 GB |
# | Classifier/Ranker | `microsoft/Phi-3.5-mini-instruct` | MIT | 3.8B | 4-bit NF4 | ~2.4 GB |
# | Writer | `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0 | 7.6B | 4-bit NF4 | ~5.5 GB |
#
# Each model is matched to how hard its job is:
#
# - **Mood/Intent → small model.** Turning one sentence into a short JSON
#   object (categories + keywords) is a narrow, highly constrained task.
#   Qwen2.5-1.5B follows JSON instructions reliably for its size and responds
#   fast, which matters because this step runs before anything else.
# - **Classifier/Ranker → mid model.** Deciding whether a story is *genuinely*
#   uplifting (not tragedy with a silver lining, not an advert) and how well it
#   fits a mood needs more judgment than intent parsing, but the output is
#   still a short JSON verdict. Phi-3.5-mini is strong at reasoning-style
#   judgments for its size. It is also a different model family from the
#   other two, so the classifier doesn't share the writer's blind spots. In
#   4-bit it takes only ~2.4 GB.
# - **Writer → large model.** Paraphrasing faithfully and hitting a 70–90 word
#   target is the hardest generation task and the one users read. Qwen2.5-7B
#   gives the best prose and instruction-following among ungated models
#   that fit on a T4. 4-bit quantization brings it from ~15 GB to ~5.5 GB.
#
# All three are ungated on Hugging Face (checked 2026-10-12), so no token is
# needed. The VRAM figures are estimates from parameter count. The smoke test
# below prints the actual peak.
#
# ### How the ModelManager works
#
# - **One interface:** `generate(model_key, messages, max_new_tokens,
#   temperature)`, where `model_key` is `"mood"`, `"classifier"` or
#   `"writer"`.
# - **Batching:** `generate_batch(model_key, list_of_messages, ...)` runs
#   several prompts in one left-padded `generate()` call. The Classifier uses
#   it to judge several stories at once. If a batch runs out of GPU memory,
#   it falls back to one prompt at a time.
# - **Prompt formatting:** each model's own chat template, via
#   `tokenizer.apply_chat_template`.
# - **Return value:** a `GenerationResult` with the text, input and output
#   token counts (from the model's real tokenizer), elapsed time, tokens per
#   second, and peak GPU memory.
# - **Lazy loading:** a model is loaded the first time it is used.
#   - `"resident"` keeps every loaded model on the GPU.
#   - `"sequential"` unloads all other models first.
#   - Unloading deletes the model, then runs `gc.collect()` and
#     `torch.cuda.empty_cache()` so the VRAM is actually released before the
#     next model loads.
#   - If resident mode runs out of GPU memory while loading or generating, the
#     manager switches itself to sequential and retries, so "Run all" never
#     needs an edit.
# - **Decoding:** `temperature=0` means greedy decoding (deterministic, used
#   for the JSON agents). `temperature > 0` samples, which the Writer uses so
#   retries produce different text.
#
# ### Mock backend
#
# When `USE_MOCK_LLM` is on, `MockModelManager` stands in with the same
# `generate` interface.
#
# - It returns canned outputs that match each agent's expected schema: valid
#   intent JSON, valid classifier JSON, and plain-text paraphrases/summaries.
#   About 1 in 5 mock stories is marked not uplifting, to exercise the
#   rejection path. The mock intent JSON comes wrapped in a code fence, as
#   real models often do, to exercise the parser.
# - Its first summary attempt is deliberately 60 words. That forces the
#   Validator to reject it, so the retry loop gets tested on CPU.
# - Token counts in mock mode are whitespace word counts, not real tokens, and
#   peak GPU memory is reported as `None`.

# %%
import gc
import json
import re
import time
import zlib
from dataclasses import dataclass

# The Writer agent puts one of these markers on the first line of its prompt (and the feedback
# marker on retries) so the mock can tell paraphrase, first summary attempt and retry apart.
TASK_PARAPHRASE = "TASK: PARAPHRASE"
TASK_SUMMARY = "TASK: SUMMARY"
SUMMARY_FEEDBACK_MARKER = "Validator feedback:"


@dataclass
class GenerationResult:
    model_key: str
    text: str
    input_token_count: int
    output_token_count: int
    elapsed_seconds: float
    tokens_per_second: float
    peak_gpu_memory_gb: float | None  # whole-process peak during this call; None on CPU/mock
    backend: str  # "hf" or "mock"


class ModelManager:
    def __init__(self, model_ids, quantize_4bit, strategy="resident", token=None):
        self.model_ids = model_ids
        self.quantize_4bit = quantize_4bit
        self.strategy = strategy
        self.token = token
        self._loaded = {}  # model_key -> (tokenizer, model)
        self.call_log = []  # every GenerationResult, in order; the orchestrator slices it per agent step

    def loaded_keys(self):
        return list(self._loaded)

    def load(self, key):
        if key in self._loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_id = self.model_ids[key]
        quant = None
        # bitsandbytes 4-bit needs CUDA; on a CPU-only debug run the model loads unquantized.
        # UNVERIFIED: 4-bit load with bitsandbytes 0.50.2 + transformers 5.17.0 on a T4 (signature checked, not run).
        if self.quantize_4bit.get(key) and HAS_CUDA:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        # fp16 matmuls are slow or unsupported on CPU, so CPU debug runs use fp32.
        dtype = getattr(torch, TORCH_DTYPE) if HAS_CUDA else torch.float32
        # Pin everything to GPU 0: device_map="auto" would silently offload layers to CPU RAM
        # instead of raising OOM, which hides the problem and makes generation extremely slow.
        device_map = {"": 0} if HAS_CUDA else {"": "cpu"}

        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device_map, quantization_config=quant, token=self.token
        )
        model.eval()
        self._loaded[key] = (tokenizer, model)
        mem = f", GPU memory allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if HAS_CUDA else ""
        print(f"Loaded {key}: {model_id} ({'4-bit' if quant else dtype}) in {time.perf_counter() - start:.1f}s{mem}")

    def unload(self, key):
        tokenizer, model = self._loaded.pop(key)
        del tokenizer, model
        gc.collect()
        if HAS_CUDA:
            torch.cuda.empty_cache()
        print(f"Unloaded {key}")

    def unload_all(self, keep=None):
        for key in [k for k in self._loaded if k != keep]:
            self.unload(key)
        # Also reclaim memory from a partially loaded model after an OOM, which is not in _loaded.
        gc.collect()
        if HAS_CUDA:
            torch.cuda.empty_cache()

    # UNVERIFIED: the OOM → sequential fallback; sequential swapping itself was tested on CPU, but no OOM was triggered.
    def _switch_to_sequential(self, key):
        print(f"CUDA out of memory with models resident; switching to sequential loading for '{key}'.")
        self.strategy = "sequential"
        self.unload_all(keep=key)

    def _ensure_loaded(self, key):
        if key in self._loaded:
            return
        if self.strategy == "sequential":
            self.unload_all()
        try:
            self.load(key)
        except torch.OutOfMemoryError:
            if self.strategy == "sequential":
                raise
            self._switch_to_sequential(key)
            self.load(key)

    def _generate_once(self, key, messages, max_new_tokens, temperature):
        tokenizer, model = self._loaded[key]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        n_in = inputs["input_ids"].shape[1]
        # Qwen's generation_config defaults to sampling, so greedy has to be requested explicitly.
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}

        if HAS_CUDA:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        if HAS_CUDA:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        new_tokens = output[0, n_in:]
        n_out = int(new_tokens.shape[0])
        return GenerationResult(
            model_key=key,
            text=tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            input_token_count=int(n_in),
            output_token_count=n_out,
            elapsed_seconds=elapsed,
            tokens_per_second=n_out / elapsed if elapsed > 0 else 0.0,
            peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 1024**3 if HAS_CUDA else None,
            backend="hf",
        )

    def generate(self, model_key, messages, max_new_tokens=256, temperature=0.0):
        self._ensure_loaded(model_key)
        try:
            result = self._generate_once(model_key, messages, max_new_tokens, temperature)
        except torch.OutOfMemoryError:
            if self.strategy == "sequential":
                raise
            self._switch_to_sequential(model_key)
            result = self._generate_once(model_key, messages, max_new_tokens, temperature)
        self.call_log.append(result)
        return result

    def _generate_batch_once(self, key, batch_messages, max_new_tokens, temperature):
        tokenizer, model = self._loaded[key]
        prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in batch_messages]
        # Decoder-only models must be left-padded so every row's generation starts right after its prompt.
        tokenizer.padding_side = "left"
        # The chat template already contains any BOS/special tokens (checked: identical ids to the
        # tokenize=True path for Qwen2.5 and Phi-3.5), so don't add them twice.
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}

        if HAS_CUDA:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        if HAS_CUDA:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1024**3 if HAS_CUDA else None

        eos = model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
        rows = []
        for i in range(len(batch_messages)):
            new_tokens = output[i, prompt_len:].tolist()
            # Finished rows are padded until the longest row ends; count only up to the first stop token.
            n_out = next((j + 1 for j, t in enumerate(new_tokens) if t in stop_ids), len(new_tokens))
            rows.append((tokenizer.decode(new_tokens[:n_out], skip_special_tokens=True).strip(),
                         int(inputs["attention_mask"][i].sum()), n_out))
        total_out = sum(n for _, _, n in rows)
        # Rows share one generate() call, so elapsed time, throughput and peak memory are per batch.
        return [
            GenerationResult(
                model_key=key, text=text, input_token_count=n_in, output_token_count=n_out,
                elapsed_seconds=elapsed, tokens_per_second=total_out / elapsed if elapsed > 0 else 0.0,
                peak_gpu_memory_gb=peak, backend="hf",
            )
            for text, n_in, n_out in rows
        ]

    def generate_batch(self, model_key, batch_messages, max_new_tokens=256, temperature=0.0):
        if not batch_messages:
            return []
        self._ensure_loaded(model_key)
        try:
            results = self._generate_batch_once(model_key, batch_messages, max_new_tokens, temperature)
            self.call_log.extend(results)
            return results
        except torch.OutOfMemoryError:
            # A batch's KV cache is the first thing to go on a tight GPU: free what we can and go one at a time.
            if self.strategy != "sequential":
                self._switch_to_sequential(model_key)
            else:
                gc.collect()
                torch.cuda.empty_cache()
            return [self.generate(model_key, m, max_new_tokens, temperature) for m in batch_messages]


_MOCK_PASSAGE = (
    "Volunteers in a small coastal town spent the spring turning an empty parking lot into a community "
    "garden that now feeds dozens of families every week. Local students built raised beds, retirees "
    "taught them how to compost, and a nearby bakery donated rain barrels to keep the plants watered "
    "through the dry months. Organizers say the project has done more than grow vegetables: neighbors "
    "who had never spoken now meet on Saturday mornings to share tools, recipes and stories. The town "
    "council has agreed to protect the site for at least ten more years, and two neighboring villages "
    "have already asked for advice on starting gardens of their own. Families who rely on the weekly "
    "harvest say fresh produce has made a real difference to their budgets and their health, and the "
    "youngest gardeners proudly show visitors the first strawberries of the season."
).split()

# Word-prefix → category hints. Used by the mock and as the rule-based fallback when the
# Mood or Classifier model output is unusable. Prefixes are chosen to avoid common false matches
# (e.g. "cats" not "cat", which would also match "catch").
CATEGORY_HINTS = {
    "hardship": "Success Through Hardship", "overcom": "Success Through Hardship", "overcam": "Success Through Hardship",
    "adversit": "Success Through Hardship", "homeless": "Success Through Hardship", "refugee": "Success Through Hardship",
    "survivor": "Success Through Hardship",
    "friend": "Friendship", "lonel": "Friendship", "companion": "Friendship", "homesick": "Friendship",
    "kindness": "Kindness", "kinder": "Kindness", "generos": "Kindness", "donat": "Kindness",
    "stranger": "Kindness", "volunteer": "Kindness", "sad": "Kindness", "griev": "Kindness", "grief": "Kindness",
    "innovat": "Innovation", "invent": "Innovation", "tech": "Innovation", "scien": "Innovation",
    "engineer": "Innovation", "robot": "Innovation", "space": "Innovation", "solar": "Innovation",
    "leader": "Leadership", "mentor": "Leadership", "captain": "Leadership", "coach": "Leadership",
    "communit": "Community Impact", "neighbo": "Community Impact", "village": "Community Impact",
    "growth": "Personal Growth", "confiden": "Personal Growth", "purpose": "Personal Growth",
    "promot": "Personal Growth", "career": "Personal Growth", "proud": "Personal Growth", "motivat": "Personal Growth",
    "persever": "Perseverance", "persist": "Perseverance", "determin": "Perseverance", "marathon": "Perseverance",
    "comeback": "Perseverance", "recover": "Perseverance", "fail": "Perseverance", "stuck": "Perseverance",
    "school": "Education", "student": "Education", "teach": "Education", "educat": "Education",
    "exam": "Education", "college": "Education", "universit": "Education", "librar": "Education", "learn": "Education",
    "health": "Health & Well-being", "stress": "Health & Well-being", "anxi": "Health & Well-being",
    "calm": "Health & Well-being", "doctor": "Health & Well-being", "hospital": "Health & Well-being",
    "medic": "Health & Well-being", "mental": "Health & Well-being", "sleep": "Health & Well-being",
    "tired": "Health & Well-being", "burn": "Health & Well-being", "overwhelm": "Health & Well-being",
    "animal": "Animals & Nature", "dog": "Animals & Nature", "pupp": "Animals & Nature", "cats": "Animals & Nature",
    "kitten": "Animals & Nature", "wildlife": "Animals & Nature", "bird": "Animals & Nature", "ocean": "Animals & Nature",
    "nature": "Animals & Nature", "forest": "Animals & Nature", "climate": "Animals & Nature", "turtle": "Animals & Nature",
    "tortoise": "Animals & Nature", "elephant": "Animals & Nature", "whale": "Animals & Nature",
}
STOPWORDS = {"i", "im", "am", "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "about", "with",
             "me", "my", "something", "feeling", "feel", "want", "some", "really", "today", "so", "is", "it",
             "be", "been", "since", "just", "very", "bit", "lot", "need", "like", "that", "this", "at", "new",
             "maybe", "kind", "sort", "would", "could", "get", "have", "has", "had"}


def categories_from_hints(text):
    cats = []
    for word in re.findall(r"[a-z]+", text.lower()):
        for hint, cat in CATEGORY_HINTS.items():
            if word.startswith(hint) and cat not in cats:
                cats.append(cat)
    return cats


def _mock_words(n):
    words = [_MOCK_PASSAGE[i % len(_MOCK_PASSAGE)] for i in range(n)]
    return " ".join(words).rstrip(".,:;") + "."


class MockModelManager:
    def __init__(self, model_ids):
        self.model_ids = model_ids
        self.call_log = []

    def loaded_keys(self):
        return []

    def unload_all(self, keep=None):
        pass

    def _respond(self, model_key, messages):
        user_text = messages[-1]["content"]
        all_text = "\n".join(m["content"] for m in messages)
        if model_key == "mood":
            quoted = re.search(r'"(.*)"', user_text, re.S)  # the mood inside the user template's quotes
            mood = quoted.group(1) if quoted else user_text
            words = [w for w in re.findall(r"[a-z]+", mood.lower()) if w not in STOPWORDS]
            # Wrapped in a code fence on purpose: real models often do this, and the parser must cope.
            return "```json\n" + json.dumps({
                "emotion": words[0] if words else "neutral",
                "desired_tone": "warm and hopeful",
                "target_categories": (categories_from_hints(mood) or CATEGORIES[:2])[:3],
                "keywords": (words or ["kindness", "community", "hope"])[:6],
            }) + "\n```"
        if model_key == "classifier":
            # crc32 rather than hash(): Python's str hash is salted per process, which would
            # make mock results change between runs.
            h = zlib.crc32(user_text.encode())
            return json.dumps({
                "category": CATEGORIES[h % (len(CATEGORIES) - 1)],
                "is_uplifting": h % 5 != 0,
                "mood_fit_score": 4 + h % 6,
                "one_line_reason": "Mock: the story's tone matches the requested mood.",
            })
        if SUMMARY_FEEDBACK_MARKER in all_text:
            return _mock_words(80)
        if TASK_SUMMARY in all_text:
            return _mock_words(60)
        return _mock_words(150)

    def generate(self, model_key, messages, max_new_tokens=256, temperature=0.0):
        start = time.perf_counter()
        text = self._respond(model_key, messages)
        result = GenerationResult(
            model_key=model_key,
            text=text,
            input_token_count=sum(len(m["content"].split()) for m in messages),
            output_token_count=len(text.split()),
            elapsed_seconds=time.perf_counter() - start,
            tokens_per_second=0.0,  # mock replies are instant, so a rate would be meaningless
            peak_gpu_memory_gb=None,
            backend="mock",
        )
        self.call_log.append(result)
        return result

    def generate_batch(self, model_key, batch_messages, max_new_tokens=256, temperature=0.0):
        return [self.generate(model_key, m, max_new_tokens, temperature) for m in batch_messages]


if USE_MOCK_LLM:
    llm = MockModelManager(MODEL_IDS)
    print("Using MockModelManager: canned outputs, no models loaded.")
else:
    llm = ModelManager(MODEL_IDS, QUANTIZE_4BIT, LOAD_STRATEGY, HF_TOKEN)
    print(f"Using ModelManager with LOAD_STRATEGY='{LOAD_STRATEGY}'. Models load on first use.")

# %% [markdown]
# ## 5. Model smoke test
#
# Sends one short prompt to each of the three models and prints the reply and
# its metrics. This proves every model loads and generates before the agents
# are built on top. On the first GPU run, this cell also downloads and loads
# the models, so it is the slow one.
#
# The table shows:
# - input and output token counts
# - generation time and tokens per second
# - peak GPU memory during the call (all models loaded at that point, not just
#   the one generating)
#
# In mock mode the replies are the canned outputs and memory shows `n/a`.
#
# **If the real models fail** to download, load or generate (for example a
# bitsandbytes/CUDA problem), this cell prints the error and switches the
# notebook to mock mode instead of stopping "Run all". The rest of the
# notebook and the UI still run, clearly labeled as mock. Fix the error and
# run everything again to use the real models.

# %%
smoke_messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer in one short sentence."},
    {"role": "user", "content": "Name one small thing that can brighten someone's day."},
]
try:
    smoke_results = [llm.generate(key, smoke_messages, max_new_tokens=40, temperature=0.0) for key in MODEL_IDS]
except Exception as e:
    if isinstance(llm, MockModelManager):
        raise
    # One failed download or 4-bit load must not stop "Run all" before the UI launches: degrade to mock mode,
    # loudly, and keep going. The UI header and every response then say "mock".
    print("=" * 72)
    print(f"ERROR: the real models failed to load or run: {type(e).__name__}: {str(e)[:300]}")
    print("Falling back to MOCK MODE so the rest of the notebook and the UI still run.")
    print("Fix the error above, then Runtime → Run all again to use the real models.")
    print("=" * 72)
    try:
        llm.unload_all()
    except Exception:
        pass
    llm = MockModelManager(MODEL_IDS)
    USE_MOCK_LLM = True
    smoke_results = [llm.generate(key, smoke_messages, max_new_tokens=40, temperature=0.0) for key in MODEL_IDS]

print(f"{'model':<11} {'in tok':>6} {'out tok':>7} {'secs':>6} {'tok/s':>6} {'peak GB':>8}  reply")
for r in smoke_results:
    peak = f"{r.peak_gpu_memory_gb:.2f}" if r.peak_gpu_memory_gb is not None else "n/a"
    reply = r.text.replace("\n", " ")
    print(f"{r.model_key:<11} {r.input_token_count:>6} {r.output_token_count:>7} {r.elapsed_seconds:>6.2f} "
          f"{r.tokens_per_second:>6.1f} {peak:>8}  {reply[:90]}{'…' if len(reply) > 90 else ''}")
print(f"\nBackend: {smoke_results[0].backend}; loaded models: {llm.loaded_keys()}")

# %% [markdown]
# ## 6. Data schemas
#
# The dataclasses passed between agents. A few helper types live with the
# code that uses them:
# - `GenerationResult`: section 4
# - `FeedStatus` and `RetrievalResult`: section 8
# - `TraceStep` and `AgentTrace`: section 15
#
# - **`TraceEvent`:** one human-readable step logged by an agent. The UI shows
#   the full trace so the multi-agent flow is visible, including every use of
#   a fallback.
# - **`MoodIntent`:** what the Mood/Intent agent extracts from the user's text:
#   emotion, desired tone, target categories and search keywords. The
#   Retrieval agent ranks stories by its keywords; the Classifier scores
#   stories against all of it. `parse_method` records how the model output
#   was read:
#   - `"json"`: valid JSON
#   - `"regex"`: fields recovered from broken JSON
#   - `"default"`: rule-based fallback
# - **`Article`:** one retrieved story, with title, source name, original URL,
#   publish date and extracted text. `text_source` records where the text came
#   from:
#   - `"extracted"`: the live page, via trafilatura
#   - `"rss_content"`: the full text included in the feed
#   - `"fallback"`: the bundled offline stories
# - **`ClassifiedStory`:** an `Article` plus the Classifier's verdict: one
#   allowed category, whether it is genuinely uplifting, a 0–10 mood-fit
#   score, and a one-line reason. It also carries the final ranking score
#   (mood fit plus a bonus if the category is one the reader asked for).
# - **`StoryDetail`:** the finished story the UI shows: title, source, URL,
#   category, paraphrase, summary and its word count, plus everything the
#   Validator found:
#   - whether the summary passed, and how (first try, after retries,
#     trimmed or extended)
#   - the copy-check numbers and the fact-check results
#   - human-readable `validation_notes`
#   - a `CallMetric` for every model call made for this story

# %%
from datetime import datetime, timezone


@dataclass
class TraceEvent:
    agent: str
    message: str
    elapsed_s: float = 0.0
    fallback_used: bool = False


@dataclass
class MoodIntent:
    raw_mood: str
    emotion: str
    desired_tone: str
    target_categories: list[str]  # 0–3 entries, each from CATEGORIES; empty means no preference
    keywords: list[str]
    parse_method: str  # "json" | "regex" | "default"


@dataclass
class Article:
    title: str
    url: str
    source_name: str
    published: datetime | None
    text: str
    text_source: str  # "extracted" | "rss_content" | "fallback"
    prefilter_score: float = 0.0


@dataclass
class ClassifiedStory:
    article: Article
    category: str  # always one of CATEGORIES
    is_uplifting: bool
    mood_fit_score: int  # 0–10
    reason: str
    category_match: bool  # category is one of the reader's target_categories
    final_score: float  # ranking key: mood fit + category bonus + small keyword tie-breaker
    parse_method: str  # "json" | "regex" | "default"


@dataclass
class CallMetric:
    purpose: str  # e.g. "paraphrase", "summary attempt 2"
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float  # for batched calls, the whole batch's time (rows run together)
    tokens_per_second: float
    peak_gpu_memory_gb: float | None
    backend: str


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
    word_count: int  # summary word count, as counted by count_words()
    summary_passed: bool  # 70 <= word_count <= 90
    summary_status: str  # "ok" | "ok_after_retry" | "trimmed" | "extended" | "out_of_range"
    summary_attempt_word_counts: list[int]
    copy_longest_ngram: int  # longest run of words shared by paraphrase and source
    copy_5gram_ratio: float  # share of the paraphrase's 5-word sequences found in the source
    copy_flagged: bool
    paraphrase_facts: dict  # fact_check() result: names/numbers checked, supported, and those missing from the source
    summary_facts: dict
    facts_flagged: bool  # a warning only: the story is still shown
    is_fallback_story: bool  # built-in placeholder story, not real news
    validation_notes: list[str]
    calls: list[CallMetric]

# %% [markdown]
# ## 7. Bundled fallback stories
#
# If live retrieval yields fewer than `MIN_USABLE_STORIES` (5) usable
# articles, for example because there is no network or the feeds block
# Colab, the Retrieval agent tops up the list from these 8 stories. The demo
# then still works end to end.
#
# - **Original placeholder text.** These are short stubs written for this
#   notebook. They are **not real news** and contain no copied article text.
# - **Clearly labeled.** The source name is "Built-in sample (offline
#   fallback)" and the URLs use the reserved `example.org` domain, so they
#   cannot be mistaken for real articles. The trace and the UI flag every
#   story that comes from here.

# %%
FALLBACK_SOURCE_NAME = "Built-in sample (offline fallback)"

FALLBACK_STORIES = [
    {
        "title": "Neighbors Turn an Empty Lot Into a Shared Vegetable Garden",
        "url": "https://example.org/fallback/shared-garden",
        "text": (
            "Residents of a quiet residential street spent three months clearing rubble from a vacant lot and "
            "turning it into a shared vegetable garden. Nobody owned a plot outright; instead, families signed "
            "up for weekly watering shifts and split the harvest. Retired neighbors taught teenagers how to "
            "start seedlings indoors, and a local hardware store lent tools for the season. By late summer the "
            "garden was producing more tomatoes, beans and squash than the street could eat, so volunteers "
            "began dropping the surplus at a nearby food pantry. Organizers say the bigger change was social: "
            "people who had lived side by side for years without speaking now gather at the garden gate most "
            "evenings. Two nearby streets have asked for help starting gardens of their own next spring."
        ),
    },
    {
        "title": "Rehabilitated Sea Turtles Head Back to the Ocean",
        "url": "https://example.org/fallback/sea-turtles",
        "text": (
            "A small coastal rescue center released a group of rehabilitated sea turtles back into the ocean "
            "this week after months of care. The turtles had been found stranded and cold-stunned during an "
            "unusually harsh winter, too weak to swim. Staff and volunteers kept them in warmed tanks, treated "
            "minor injuries and slowly built up their strength with a diet of fish and greens. Before release, "
            "each turtle passed a health check and was fitted with a small tag so researchers can learn where "
            "it travels. Dozens of local schoolchildren lined the beach to watch the animals crawl into the "
            "surf. The center says this season's survival rate was the highest it has recorded, thanks largely "
            "to new volunteers who answered an appeal for help."
        ),
    },
    {
        "title": "Students Help Astronomers Confirm a Changing Star",
        "url": "https://example.org/fallback/student-astronomers",
        "text": (
            "A group of high school students working with a university outreach program helped confirm that "
            "a faint star changes brightness on a regular cycle. The students used a remotely operated "
            "telescope to take images every clear night for two months, then measured the star's brightness "
            "in each frame with free software. Their measurements matched those from professional "
            "observatories closely enough that the data were included in a shared research database. "
            "Teachers say the project changed how students see science: instead of reading about discoveries, "
            "they took part in one. Several participants now plan to study physics or engineering, and the "
            "program is expanding to include three more schools next year."
        ),
    },
    {
        "title": "A Morning Walking Club Helps Older Residents Feel Less Alone",
        "url": "https://example.org/fallback/walking-club",
        "text": (
            "What began as two retired friends walking laps around a park has grown into a morning walking "
            "club with more than forty regular members. The group meets at the same bench at eight o'clock "
            "every weekday and walks at a gentle pace, so anyone can join regardless of fitness. A volunteer "
            "nurse checks in once a month to answer health questions, and members take turns phoning anyone "
            "who misses more than two walks in a row. Several members say they had barely left home before "
            "joining and now look forward to the mornings. Local doctors have started recommending the club "
            "to patients who live alone, describing regular company and light exercise as a simple boost for "
            "both body and mood."
        ),
    },
    {
        "title": "Fish Return to a River After Years of Volunteer Cleanups",
        "url": "https://example.org/fallback/river-cleanup",
        "text": (
            "A river that was once choked with litter and runoff is showing strong signs of recovery after "
            "years of volunteer cleanups. Every spring, residents have hauled out shopping carts, tires and "
            "plastic, while local farmers planted strips of native plants along the banks to filter rainwater "
            "before it reaches the water. This year, survey teams counted several fish species that had not "
            "been seen in the river for decades, along with returning herons and otters. Water quality tests "
            "show oxygen levels at their healthiest since monitoring began. Volunteers say the turnaround "
            "proves that steady local effort adds up, and they are now working with schools to train the next "
            "generation of river keepers."
        ),
    },
    {
        "title": "Teen Volunteers Help Young Children Fall in Love With Reading",
        "url": "https://example.org/fallback/reading-buddies",
        "text": (
            "A public library has paired teenage volunteers with young children who are struggling to read, "
            "and the results have delighted parents and teachers. Twice a week, each pair spends half an hour "
            "reading picture books together, taking turns page by page. The teenagers receive a short training "
            "session on patience and encouragement, and they earn community service hours in return. After "
            "one school year, most of the younger readers had moved up at least one reading level, and many "
            "now ask to borrow books on their own. The teen volunteers say they gained something too: several "
            "described feeling trusted and needed for the first time. The library plans to double the "
            "program's size this autumn."
        ),
    },
    {
        "title": "Local Painters Brighten Hospital Corridors With Murals",
        "url": "https://example.org/fallback/hospital-murals",
        "text": (
            "Long, beige hospital corridors have been transformed into bright landscapes by a team of local "
            "artists who volunteered their weekends. The murals show forests, beaches and night skies, chosen "
            "after staff asked patients what scenes they found most calming. Painters worked in short shifts "
            "so as not to disturb wards, and young patients were invited to add small details such as birds "
            "and stars. Nurses report that children now ask to walk to their appointments instead of being "
            "carried, and families say the waiting areas feel less frightening. The artists donated their "
            "time, while a neighborhood fundraiser covered the paint. Another hospital wing has already "
            "requested the same treatment."
        ),
    },
    {
        "title": "Bus Driver Helps a Lost Child Find Her Way Home",
        "url": "https://example.org/fallback/bus-driver",
        "text": (
            "A city bus driver is being thanked by a grateful family after helping a young girl who had become "
            "separated from her grandmother at a busy market. The driver noticed the child crying alone at a "
            "stop, pulled over safely and kept her company while calling the transit control center. Rather "
            "than leaving her with strangers, he waited with her, chatting about her favorite cartoons to keep "
            "her calm, until police and her worried grandmother arrived about twenty minutes later. Passengers "
            "on the bus said nobody minded the delay, and several applauded when the two were reunited. The "
            "transit agency has given the driver a commendation, and the family delivered a homemade cake to "
            "the depot."
        ),
    },
]

# %% [markdown]
# ## 8. Agent 2: Retrieval (non-LLM tool)
#
# Finds candidate stories that match the user's mood. It uses no LLM; it is
# a deterministic tool.
#
# 1. **Fetch feeds in parallel.** It reads every feed in `RSS_SOURCES` with
#    `requests` and parses the result with `feedparser`.
#    - Each request has a connect/read timeout and up to `HTTP_RETRIES`
#      retries with backoff, covering connection errors, HTTP 429 and 5xx.
#    - Requests send a User-Agent that names the project and identifies it as
#      an educational demo, and at most 2 requests go to any one site at a
#      time. Some pages behind bot protection reject this honest User-Agent;
#      they are dropped and logged rather than worked around.
#    - A feed that fails is logged and skipped; it never stops the others.
#    - Parsed feeds, and the article fetches in step 4, are cached for 15
#      minutes, so repeat UI requests don't hammer the sites.
# 2. **Deduplicate** by normalized URL (tracking parameters like `utm_*`
#    stripped) and by normalized title.
# 3. **Rank** each entry by keyword overlap with the `MoodIntent`. A match in
#    the title counts double; ties go to newer stories. If the user named a
#    preferred source, its stories rank first, and the other sources fill any
#    remaining slots. The name is matched against the source names in the
#    config, tolerating partial names and small typos.
# 4. **Fetch and extract** the top `2 × PREFILTER_K` articles in parallel.
#    - Fetching each page also checks that the URL actually resolves. Pages
#      with HTTP errors or non-HTML content are dropped, and the final URL
#      after redirects is kept as the canonical link.
#    - trafilatura extracts the main text and a publish date if the feed
#      lacks one. If extraction comes back short, the full text included in
#      the feed is used when there is one.
#    - Articles under `MIN_ARTICLE_CHARS` are dropped. Text is cut to
#      `MAX_ARTICLE_CHARS` at a word boundary.
# 5. **Fallback.** If fewer than `MIN_USABLE_STORIES` live articles survive,
#    bundled stories from section 7 fill the list up to `PREFILTER_K`. This
#    is logged to the trace with `fallback_used=True`.
#
# Every step, including each dropped article and the reason, is recorded as a
# `TraceEvent`.

# %%
import difflib
import html as html_lib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import requests
import trafilatura
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class FeedStatus:
    source_name: str
    url: str
    ok: bool
    n_entries: int
    elapsed_s: float
    from_cache: bool = False
    error: str | None = None


@dataclass
class _Candidate:
    title: str
    url: str
    source_name: str
    published: datetime | None
    summary: str
    rss_content: str
    score: float = 0.0


@dataclass
class RetrievalResult:
    articles: list[Article]
    used_fallback: bool
    feed_statuses: list[FeedStatus]
    preferred_source: str | None  # the config source name it matched, if any
    urls_attempted: int = 0  # shortlisted article URLs fetched (cache hits included)
    urls_resolved: int = 0  # of those, how many answered with a usable page (HTTP < 400), even if later dropped


def _make_session():
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,  # hand the final response back so the status code can be reported
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": HTTP_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return session


HTTP = _make_session()
_HTTP_TIMEOUT = (5, HTTP_TIMEOUT_S)  # (connect, read)
# A preferred source can put the whole shortlist on one host; cap parallel requests per host to stay polite.
_MAX_REQUESTS_PER_HOST = 2
_host_slots: dict[str, threading.BoundedSemaphore] = {}
_host_slots_lock = threading.Lock()
_feed_cache: dict[str, tuple[float, list]] = {}
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")


def _strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    # Sunny Skyz's feed ships "and#039;" where "&#039;" was meant (the "&" was replaced with "and").
    text = re.sub(r"and#(\d+);", r"&#\1;", text)
    return re.sub(r"\s+", " ", html_lib.unescape(text)).strip()


def _url_key(url):
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith(_TRACKING_PARAMS)])
    host = parts.netloc.lower().removeprefix("www.")
    return urlunsplit(("", host, parts.path.rstrip("/"), query, ""))


def _title_key(title):
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _entry_datetime(entry):
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    # feedparser normalizes dates to UTC struct_time.
    return datetime(*parsed[:6], tzinfo=timezone.utc) if parsed else None


def _parse_iso_date(value):
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc) if value else None
    except ValueError:
        return None


def _truncate(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind(" ")] if " " in cut else cut


def keyword_score(keywords, title, summary):
    title_l, summary_l = title.lower(), summary.lower()
    score = 0.0
    for kw in keywords:
        # Prefix match ("animal" also matches "animals"), anchored at a word start.
        pattern = r"\b" + re.escape(kw.lower().strip())
        if re.search(pattern, title_l):
            score += 2
        elif re.search(pattern, summary_l):
            score += 1
    return score


def match_source(preferred, sources=None):
    sources = sources or RSS_SOURCES
    if not preferred or not preferred.strip():
        return None
    wanted = preferred.strip().lower()
    for name in sources:
        if wanted == name.lower() or wanted in name.lower() or name.lower() in wanted:
            return name
    by_lower = {name.lower(): name for name in sources}
    close = difflib.get_close_matches(wanted, list(by_lower), n=1, cutoff=0.6)
    return by_lower[close[0]] if close else None


def fetch_feed(name, url):
    cached = _feed_cache.get(url)
    if cached and time.time() - cached[0] < FEED_CACHE_TTL_S:
        return cached[1], FeedStatus(name, url, True, len(cached[1]), 0.0, from_cache=True)
    start = time.perf_counter()
    try:
        resp = HTTP.get(url, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        if not parsed.entries:
            raise ValueError(f"no entries in feed ({parsed.get('bozo_exception', 'empty')})")
    except Exception as e:
        # Network errors, HTTP errors and malformed XML all mean the same thing here: skip this feed.
        return [], FeedStatus(name, url, False, 0, time.perf_counter() - start, error=f"{type(e).__name__}: {e}"[:200])
    _feed_cache[url] = (time.time(), parsed.entries)
    return parsed.entries, FeedStatus(name, url, True, len(parsed.entries), time.perf_counter() - start)


def check_feeds(sources=None):
    sources = sources or RSS_SOURCES
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        results = list(pool.map(lambda item: fetch_feed(*item), sources.items()))
    statuses = [status for _, status in results]
    print(f"{'source':<24} {'status':<8} {'entries':>7} {'secs':>5}  detail")
    for s in statuses:
        detail = "cached" if s.from_cache else (s.error or s.url)
        print(f"{s.source_name:<24} {'OK' if s.ok else 'FAILED':<8} {s.n_entries:>7} {s.elapsed_s:>5.1f}  {detail}")
    print(f"\n{sum(s.ok for s in statuses)}/{len(statuses)} feeds responded.")
    return statuses


def _to_candidate(entry, source_name):
    content = entry.get("content") or []
    return _Candidate(
        title=_strip_html(entry.get("title", "")),
        url=entry.get("link", ""),
        source_name=source_name,
        published=_entry_datetime(entry),
        summary=_strip_html(entry.get("summary", "")),
        rss_content=_strip_html(content[0].get("value", "")) if content else "",
    )


def _host_slot(url):
    host = urlsplit(url).netloc.lower()
    with _host_slots_lock:
        return _host_slots.setdefault(host, threading.BoundedSemaphore(_MAX_REQUESTS_PER_HOST))


def fetch_article(cand):
    try:
        with _host_slot(cand.url):
            resp = HTTP.get(cand.url, timeout=_HTTP_TIMEOUT)
    except requests.RequestException as e:
        return None, f"URL did not resolve ({type(e).__name__})"
    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}"
    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type:
        return None, f"not an HTML page ({content_type or 'unknown type'})"

    # Pass bytes, not resp.text, so trafilatura detects the encoding from the page itself.
    doc = trafilatura.bare_extraction(
        resp.content, url=resp.url, with_metadata=True,
        include_comments=False, include_tables=False, favor_precision=True,
    )
    text, text_source = ((doc.text if doc else None) or "").strip(), "extracted"
    if len(text) < MIN_ARTICLE_CHARS and len(cand.rss_content) >= MIN_ARTICLE_CHARS:
        text, text_source = cand.rss_content, "rss_content"
    if len(text) < MIN_ARTICLE_CHARS:
        return None, f"too short ({len(text)} chars < {MIN_ARTICLE_CHARS})"

    return Article(
        title=cand.title,
        url=resp.url,
        source_name=cand.source_name,
        published=cand.published or _parse_iso_date(doc.date if doc else None),
        text=_truncate(text, MAX_ARTICLE_CHARS),
        text_source=text_source,
        prefilter_score=cand.score,
    ), None


_article_cache: dict[str, tuple[float, Article | None, str | None]] = {}
ARTICLE_CACHE_STATS = {"hits": 0, "misses": 0}  # cumulative; the orchestrator reports per-request deltas
_article_stats_lock = threading.Lock()  # fetch_article_cached runs in worker threads


def fetch_article_cached(cand):
    cached = _article_cache.get(cand.url)
    hit = bool(cached and time.time() - cached[0] < FEED_CACHE_TTL_S)
    with _article_stats_lock:
        ARTICLE_CACHE_STATS["hits" if hit else "misses"] += 1
    if hit:
        article, reason = cached[1], cached[2]
    else:
        article, reason = fetch_article(cand)
        _article_cache[cand.url] = (time.time(), article, reason)
    # The keyword score depends on the current mood, so it is re-applied to cached articles.
    return (replace(article, prefilter_score=cand.score) if article else None), reason


def _fallback_articles(keywords, n):
    stories = [
        Article(
            title=s["title"], url=s["url"], source_name=FALLBACK_SOURCE_NAME, published=None,
            text=s["text"], text_source="fallback", prefilter_score=keyword_score(keywords, s["title"], s["text"]),
        )
        for s in FALLBACK_STORIES
    ]
    stories.sort(key=lambda a: -a.prefilter_score)
    return stories[:n]


def retrieve_stories(intent, preferred_source=None, trace=None, n_target=PREFILTER_K, sources=None):
    sources = sources or RSS_SOURCES
    trace = trace if trace is not None else []
    start = time.perf_counter()

    def log(message, fallback=False):
        trace.append(TraceEvent("retrieval", message, round(time.perf_counter() - start, 2), fallback))

    preferred = match_source(preferred_source, sources)
    if preferred_source and not preferred:
        log(f"Preferred source '{preferred_source}' matches none of: {', '.join(sources)}. Using all sources.")
    elif preferred:
        log(f"Preferred source: {preferred} (its stories are ranked first).")

    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        feed_results = list(pool.map(lambda item: fetch_feed(*item), sources.items()))
    statuses = [status for _, status in feed_results]
    for s in statuses:
        if s.ok:
            log(f"Feed OK: {s.source_name} ({s.n_entries} entries{', cached' if s.from_cache else ''})")
        else:
            log(f"Feed FAILED: {s.source_name}: {s.error}")

    seen_urls, seen_titles, candidates = set(), set(), []
    for entries, status in feed_results:
        for entry in entries:
            cand = _to_candidate(entry, status.source_name)
            if not cand.url or not cand.title:
                continue
            url_key, title_key = _url_key(cand.url), _title_key(cand.title)
            if url_key in seen_urls or title_key in seen_titles:
                continue
            seen_urls.add(url_key)
            seen_titles.add(title_key)
            cand.score = keyword_score(intent.keywords, cand.title, cand.summary)
            candidates.append(cand)

    candidates.sort(key=lambda c: (
        c.source_name != preferred,
        -c.score,
        -(c.published.timestamp() if c.published else 0),
    ))
    # Fetch extra so that dropped (dead, short, non-HTML) articles don't leave us under target.
    shortlist = candidates[: n_target * 2]
    log(f"{len(candidates)} unique entries after dedupe; fetching the top {len(shortlist)} by keyword match "
        f"(keywords: {', '.join(intent.keywords)}).")

    articles, final_urls, urls_resolved = [], set(), 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for cand, (article, reason) in zip(shortlist, pool.map(fetch_article_cached, shortlist)):
            # fetch_article's failure reasons start with one of these when the URL itself didn't work.
            if article is not None or not reason.startswith(("URL did not resolve", "HTTP ")):
                urls_resolved += 1
            if article is None:
                log(f"Dropped '{cand.title[:70]}' ({cand.source_name}): {reason}")
            elif _url_key(article.url) in final_urls:
                # Two feed links can redirect to the same article.
                log(f"Dropped '{cand.title[:70]}' ({cand.source_name}): duplicate after redirect")
            else:
                final_urls.add(_url_key(article.url))
                articles.append(article)
    articles = articles[:n_target]
    n_live = len(articles)

    used_fallback = n_live < MIN_USABLE_STORIES
    if used_fallback:
        extra = _fallback_articles(intent.keywords, n_target - n_live)
        articles.extend(extra)
        log(f"FALLBACK USED: only {n_live} usable live stories (< {MIN_USABLE_STORIES}); added {len(extra)} "
            f"built-in placeholder stories marked '{FALLBACK_SOURCE_NAME}'.", fallback=True)

    log(f"Returning {len(articles)} articles ({n_live} live, {len(articles) - n_live} fallback).",
        fallback=used_fallback)
    return RetrievalResult(articles, used_fallback, statuses, preferred, len(shortlist), urls_resolved)

# %% [markdown]
# ### Feed health check
#
# Runs once at startup and reports which feeds responded, with entry counts,
# timing and the error for any that failed. If every feed fails (for example,
# a site blocks Colab's IP range), the pipeline still works using the
# bundled fallback stories, and the UI says so.

# %%
feed_statuses = check_feeds()

# %% [markdown]
# ### Retrieval test
#
# Runs the Retrieval agent with a hand-written `MoodIntent`, since the Mood
# agent comes in the next section. It prints:
# - each candidate's source, publish date, text length, where the text came
#   from, and keyword score
# - the agent's trace
#
# A second run uses a feed URL that can never resolve (the reserved
# `.invalid` domain). This shows the fallback path: the stories are marked as
# built-in samples and the trace says `FALLBACK USED`.

# %%
def print_retrieval(result, trace):
    print(f"{'#':>2} {'source':<24} {'published':<10} {'chars':>5} {'text from':<11} {'score':>5}  title / url")
    for i, a in enumerate(result.articles, 1):
        date = a.published.strftime("%Y-%m-%d") if a.published else "—"
        print(f"{i:>2} {a.source_name[:24]:<24} {date:<10} {len(a.text):>5} {a.text_source:<11} "
              f"{a.prefilter_score:>5.1f}  {a.title[:70]}")
        print(f"{'':>67}{a.url}")
    print(f"\nused_fallback={result.used_fallback}  preferred_source={result.preferred_source}")
    print("Trace:")
    for ev in trace:
        print(f"  [{ev.elapsed_s:>5.2f}s]{' (fallback)' if ev.fallback_used else ''} {ev.message}")


test_intent = MoodIntent(
    raw_mood="a bit stressed, want something about animals or nature",
    emotion="stressed",
    desired_tone="calm and gentle",
    target_categories=["Animals & Nature"],
    keywords=["animal", "wildlife", "rescue", "nature", "ocean", "dog"],
    parse_method="default",
)
live_trace = []
live_result = retrieve_stories(test_intent, preferred_source="good news", trace=live_trace)
print_retrieval(live_result, live_trace)

print("\n" + "=" * 100 + "\nFallback demo (unreachable feed):\n")
fallback_trace = []
fallback_result = retrieve_stories(
    test_intent, trace=fallback_trace, sources={"Unreachable test feed": "https://feed.invalid/rss"}
)
print_retrieval(fallback_result, fallback_trace)

# %% [markdown]
# ## 9. Robust JSON parsing
#
# Small open models usually return clean JSON, but not always. Common
# failures are wrapping the JSON in a ```` ```json ```` fence, adding chatter
# before or after it ("Sure! Here is…"), using single quotes or Python's
# `True`, leaving a trailing comma, or getting cut off by the token limit.
# `parse_json_object` tries these in order and returns `None` instead of
# raising:
#
# 1. Strip a code fence if there is one.
# 2. Find the first *balanced* `{...}` object. The scan tracks strings, so a
#    brace inside a string doesn't confuse it.
# 3. Try `json.loads`.
# 4. Repair and retry: remove trailing commas, swap curly quotes for straight
#    ones, then try it as a Python literal so single quotes and
#    `True`/`False`/`None` work.
#
# If all of that fails, `regex_extract` pulls individual fields
# (`"key": value`) out of the raw text. The agents then validate every field
# and fill anything still missing with a sensible default. **No model output
# can raise an exception past this point.**

# %%
import ast

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def _first_json_object(text):
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # unbalanced, e.g. cut off by max_new_tokens; regex_extract handles this case


def _loads_lenient(candidate):
    attempts = [candidate]
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    attempts.append(repaired)
    for attempt in attempts:
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    # Single-quoted keys/values and Python booleans are valid Python literals but not JSON.
    pythonish = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", repaired)))
    try:
        obj = ast.literal_eval(pythonish)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def parse_json_object(text):
    text = (text or "").strip()
    if not text:
        return None
    sources = []
    fence = _FENCE_RE.search(text)
    if fence:
        sources.append(fence.group(1))
    sources.append(text)
    for source in sources:
        obj_text = _first_json_object(source)
        if obj_text:
            parsed = _loads_lenient(obj_text)
            if parsed is not None:
                return parsed
    return None


def regex_extract(text, key, kind):
    text = text or ""
    k = re.escape(key)
    if kind == "str":
        m = re.search(rf"""["']?{k}["']?\s*[:=]\s*["']([^"'\n]+)""", text, re.I)
        return m.group(1).strip() if m else None
    if kind == "num":
        m = re.search(rf"""["']?{k}["']?\s*[:=]\s*["']?(-?\d+(?:\.\d+)?)""", text, re.I)
        return float(m.group(1)) if m else None
    if kind == "bool":
        m = re.search(rf"""["']?{k}["']?\s*[:=]\s*["']?(true|false|yes|no)\b""", text, re.I)
        return m.group(1).lower() in ("true", "yes") if m else None
    if kind == "list":
        # Also accepts a list cut off before its closing bracket.
        m = re.search(rf"""["']?{k}["']?\s*[:=]\s*\[([^\]]*)""", text, re.I)
        return re.findall(r"""["']([^"'\n]+)["']""", m.group(1)) if m else None
    raise ValueError(f"unknown kind {kind!r}")


def parse_model_output(text, fields):
    """Return (dict, method) where method is "json", "regex" or "default" (nothing recovered)."""
    parsed = parse_json_object(text)
    if parsed is not None:
        return parsed, "json"
    recovered = {key: regex_extract(text, key, kind) for key, kind in fields.items()}
    recovered = {k: v for k, v in recovered.items() if v is not None}
    return (recovered, "regex") if recovered else ({}, "default")


def normalize_category(value):
    if not isinstance(value, str) or not value.strip():
        return None
    wanted = value.strip().lower()
    by_lower = {c.lower(): c for c in CATEGORIES}
    if wanted in by_lower:
        return by_lower[wanted]
    for lower, original in by_lower.items():
        # "Animals" -> "Animals & Nature"; "Health" -> "Health & Well-being"; "Kindness story" -> "Kindness"
        if wanted in lower or lower in wanted:
            return original
    close = difflib.get_close_matches(wanted, list(by_lower), n=1, cutoff=0.6)
    return by_lower[close[0]] if close else None


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "y", "1"):
            return True
        if v in ("false", "no", "n", "0"):
            return False
    return None


def to_score(value, low=0, high=10):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)  # "7/10", "8 out of 10", "score: 6.5"
        value = float(m.group()) if m else None
    if isinstance(value, (int, float)):
        return int(max(low, min(high, round(value))))
    return None

# %% [markdown]
# ## 10. Prompt design
#
# Both LLM agents share the same approach, built for small models that need
# to produce machine-readable output:
#
# - **System prompt = role + exact output shape + field rules.** The first
#   thing the model sees is "reply with ONLY a JSON object", followed by a
#   one-line template of that object with `<...>` placeholders. Concrete
#   example values would bias small models toward copying them, which is
#   why the template uses placeholders.
# - **Closed vocabularies are listed verbatim.** The allowed categories are
#   inserted from `CATEGORIES` as a bullet list, and the model is told to copy
#   one exactly. The code still checks: near misses ("Animals") are mapped
#   to the closest allowed name, and unknown values fall back to keyword
#   hints or "Other Uplifting".
# - **One worked example for the Mood agent.** A user/assistant example pair
#   (on a different mood from the test moods) shows the expected style. That
#   matters most for the keywords, which should be *concrete headline words*
#   ("rescue", "volunteers", "puppies"), not feelings. Feelings like "stress"
#   would match bad-news headlines in the Retrieval agent's keyword ranking.
#   The code also drops negative feeling words from the keyword list in case
#   the model ignores the rule. The Classifier gets no example: it runs once
#   per story, so every extra prompt token is paid 8 times.
# - **An explicit definition of "genuinely uplifting".** The Classifier is
#   told what does *not* count: tragedy with a silver lining, crime,
#   disaster, adverts, fundraising appeals, podcast listings. Positive-news
#   feeds include all of these.
# - **The Classifier never sees the reader's preferred categories.** When
#   they were in the prompt, a test model copied a preferred category onto
#   unrelated stories: a sea-turtle release and a deadly fire were both
#   given the reader's preferred category. A category should describe the
#   story alone. The
#   preference is applied afterwards in code, as the ranking bonus.
# - **An anchored 0–10 scale.** Mood fit is defined at 10 / 7 / 4 / 0, so
#   scores are comparable across stories and runs.
# - **Short, bounded inputs.** Mood text is cut to `MAX_MOOD_CHARS`. Each
#   story is sent as title, source and the first `CLASSIFIER_EXCERPT_CHARS`
#   characters (~350 tokens), cut at a word boundary. That is enough to judge
#   category and tone, and it keeps each prompt to well under 1k tokens,
#   inside every model's context window.
# - **Greedy decoding** (`temperature=0`) for both agents: the output is a
#   judgment, so it should be deterministic.
# - **Batching:** Classifier prompts go through `llm.generate_batch` in groups
#   of `CLASSIFIER_BATCH_SIZE`, which is much faster on a GPU than one story
#   at a time.
# - **One repair retry:** if a Classifier reply can't be parsed as JSON, the
#   model is shown its own reply and asked once more for "only the JSON
#   object". After that, the regex and default fallbacks take over. Each step
#   is logged to the trace.
#
# **Ranking.** Each story's `final_score` is:
#
# - `mood_fit_score`
# - plus `CATEGORY_MATCH_BONUS` (2) if its category is one of the reader's
#   target categories
# - plus a small tie-breaker from the Retrieval keyword score (at most 0.5)
#
# Stories judged not uplifting are removed, never padded back in. The top
# `CLASSIFIER_TOP_N` (5) go to the Writer. If fewer than 5 live stories
# pass, the orchestrator (section 15) tops up with labeled built-in stories,
# judged the same way.

# %%
MOOD_SYSTEM_PROMPT = """You help an uplifting-news app choose stories for a reader based on how they feel.
Read the reader's message and reply with ONLY a JSON object, no other text, in exactly this format:
{{"emotion": "<one or two words>", "desired_tone": "<short phrase>", "target_categories": ["<category>", ...], "keywords": ["<word>", ...]}}

Rules:
- "emotion": the reader's current emotion in one or two words, e.g. "stressed", "lonely", "bored".
- "desired_tone": a short phrase describing the kind of story that would help them, e.g. "calm and reassuring".
- "target_categories": 1 to 3 categories, each copied exactly from this list:
{category_list}
- "keywords": 4 to 8 lowercase single words likely to appear in the HEADLINE of an uplifting story that would help this reader: topics, people, animals or actions such as "rescue", "students", "volunteers", "puppies". Never use negative feeling words such as "stress", "sad" or "anxiety"; they match bad news."""

MOOD_USER_TEMPLATE = 'Reader\'s message: "{mood}"'

MOOD_FEWSHOT = [
    {"role": "user", "content": MOOD_USER_TEMPLATE.format(mood="I miss my grandmother and the house feels too quiet")},
    {"role": "assistant", "content": json.dumps({
        "emotion": "lonely",
        "desired_tone": "warm and comforting",
        "target_categories": ["Friendship", "Kindness"],
        "keywords": ["grandmother", "neighbors", "friendship", "volunteers", "reunion", "community"],
    })},
]

CLASSIFIER_SYSTEM_PROMPT = """You are an editor for an uplifting-news app. You judge ONE news story for ONE reader.
Reply with ONLY a JSON object, no other text, in exactly this format:
{{"category": "<category>", "is_uplifting": <true or false>, "mood_fit_score": <integer 0-10>, "one_line_reason": "<one sentence>"}}

Rules:
- "category": the single best category for the story, copied exactly from this list:
{category_list}
- "is_uplifting": true only if the story's main outcome is genuinely positive or hopeful (kindness, recovery, discovery, progress, achievement). false if it is mainly about tragedy, crime, disaster, conflict or illness even with a silver lining, or if it is mainly an advert, a fundraising appeal, a product promotion or a podcast/episode listing.
- "mood_fit_score": how well this story suits this reader right now. 10 = exactly what they need; 7 = a good fit; 4 = uplifting but unrelated to their mood; 0 = likely to make them feel worse.
- "one_line_reason": one short sentence explaining the score."""

CLASSIFIER_USER_TEMPLATE = """Reader: feeling {emotion}; wants stories that feel {desired_tone}. In their words: "{raw_mood}"

Story title: {title}
Source: {source}
Story text (may be cut off):
{excerpt}"""

JSON_RETRY_PROMPT = "That reply was not valid JSON in the required format. Reply again with ONLY the JSON object, starting with { and ending with }."

_CATEGORY_LIST = "\n".join(f"  - {c}" for c in CATEGORIES)
MOOD_SYSTEM = MOOD_SYSTEM_PROMPT.format(category_list=_CATEGORY_LIST)
CLASSIFIER_SYSTEM = CLASSIFIER_SYSTEM_PROMPT.format(category_list=_CATEGORY_LIST)

# Fail fast at notebook load (not at request time) if CATEGORIES is edited without updating the example.
_bad = [c for c in json.loads(MOOD_FEWSHOT[1]["content"])["target_categories"] if c not in CATEGORIES]
assert not _bad, f"MOOD_FEWSHOT uses categories not in CATEGORIES: {_bad}"

MOOD_FIELDS = {"emotion": "str", "desired_tone": "str", "target_categories": "list", "keywords": "list"}
CLASSIFIER_FIELDS = {"category": "str", "is_uplifting": "bool", "mood_fit_score": "num", "one_line_reason": "str"}
NEGATIVE_WORDS = {"stress", "stressed", "stressful", "sad", "sadness", "anxiety", "anxious", "depressed",
                  "depression", "lonely", "loneliness", "angry", "anger", "tired", "bored", "boredom", "worried",
                  "worry", "upset", "fear", "scared", "grief", "hopeless", "overwhelmed", "frustrated"}

# %% [markdown]
# ## 11. Agent 1: Mood/Intent (small LLM)
#
# `run_mood_agent(mood_text)` turns the reader's free text into a validated
# `MoodIntent`:
#
# 1. **Prepare the input.** Whitespace is normalized and the text is cut to
#    `MAX_MOOD_CHARS`. Empty input skips the model and returns a general
#    "anything uplifting" intent.
# 2. **Call the small model.** The system prompt, one worked example and the
#    mood go to `"mood"` (Qwen2.5-1.5B), greedy, ≤160 new tokens.
# 3. **Parse** with `parse_model_output`: JSON first, then the regex field
#    recovery.
# 4. **Validate every field:**
#    - **Categories** are mapped onto `CATEGORIES`, deduplicated and capped
#      at 3. If none survive, they are guessed from words in the mood using
#      `CATEGORY_HINTS`.
#    - **Keywords** are lowercased and split into single words. Stopwords
#      and negative feeling words are dropped, and the list is capped at 8.
#      If fewer than 3 remain, it is topped up from the mood's own content
#      words and the target categories' names.
#    - **Emotion and tone** default to "unspecified" and "warm and hopeful".
#
# If the model call itself raises an error, for example running out of GPU
# memory, it is logged and the agent continues with the rule-based fallback.
# A bad model reply can lower quality but cannot stop the pipeline.

# %%
def _content_words(text):
    words = re.findall(r"[a-z]+", (text or "").lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS and w not in NEGATIVE_WORDS]


def _clean_keywords(raw_keywords, mood, categories):
    words = []
    for item in raw_keywords if isinstance(raw_keywords, list) else []:
        if isinstance(item, str):
            words.extend(_content_words(item))
    if len(set(words)) < 3:
        words += _content_words(mood) + _content_words(" ".join(categories))
    keywords = list(dict.fromkeys(words))[:8]
    return keywords or ["kindness", "community", "hope"]


def build_mood_messages(mood):
    return [{"role": "system", "content": MOOD_SYSTEM}, *MOOD_FEWSHOT,
            {"role": "user", "content": MOOD_USER_TEMPLATE.format(mood=mood)}]


def run_mood_agent(mood_text, trace=None):
    trace = trace if trace is not None else []
    start = time.perf_counter()

    def log(message, fallback=False):
        trace.append(TraceEvent("mood", message, round(time.perf_counter() - start, 2), fallback))

    mood = " ".join((mood_text or "").split())[:MAX_MOOD_CHARS]
    if not mood:
        log("Empty mood text; using a general 'anything uplifting' intent.", fallback=True)
        return MoodIntent("", "unspecified", "warm and hopeful", [], ["kindness", "community", "hope"], "default")

    text = ""
    try:
        result = llm.generate("mood", build_mood_messages(mood), max_new_tokens=160, temperature=0.0)
        text = result.text
        log(f"Model reply ({result.output_token_count} tokens, {result.elapsed_seconds:.1f}s): {text[:200]}")
    except Exception as e:
        # Any generation failure (OOM, load error) degrades to the rule-based intent instead of failing the request.
        log(f"Model call failed ({type(e).__name__}: {str(e)[:120]}); using rule-based intent.", fallback=True)

    raw, method = parse_model_output(text, MOOD_FIELDS)

    raw_categories = raw.get("target_categories")
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]
    categories = []
    for c in raw_categories if isinstance(raw_categories, list) else []:
        name = normalize_category(c)
        if name and name not in categories:
            categories.append(name)
    if not categories:
        categories = categories_from_hints(mood)
        if categories:
            log(f"No valid categories from model; guessed from mood words: {categories}", fallback=True)
    categories = categories[:3]

    emotion, tone = raw.get("emotion"), raw.get("desired_tone")
    intent = MoodIntent(
        raw_mood=mood,
        emotion=emotion.strip()[:40] if isinstance(emotion, str) and emotion.strip() else "unspecified",
        desired_tone=tone.strip()[:80] if isinstance(tone, str) and tone.strip() else "warm and hopeful",
        target_categories=categories,
        keywords=_clean_keywords(raw.get("keywords"), mood, categories),
        parse_method=method,
    )
    log(f"Intent ({method}): emotion={intent.emotion!r}, tone={intent.desired_tone!r}, "
        f"categories={intent.target_categories}, keywords={intent.keywords}", fallback=method != "json")
    return intent

# %% [markdown]
# ## 12. Agent 3: Classifier/Ranker (mid LLM)
#
# `classify_stories(intent, articles)` judges every retrieved article, and
# `rank_stories(classified)` picks the best ones.
#
# 1. **Build prompts.** Each article gets a prompt with the reader's intent
#    plus the story's title, source and an excerpt of up to
#    `CLASSIFIER_EXCERPT_CHARS`.
# 2. **Run in batches.** Prompts go to `"classifier"` (Phi-3.5-mini) in
#    batches of `CLASSIFIER_BATCH_SIZE`, greedy, ≤120 new tokens each. If a
#    batch fails outright, its stories get empty replies and fall through to
#    the defaults below.
# 3. **Retry unparsable replies once** in a single batch, using the repair
#    prompt.
# 4. **Validate each verdict:**
#    - **Category:** must map onto `CATEGORIES`. Otherwise it is guessed from
#      words in the title using `CATEGORY_HINTS`, else set to "Other
#      Uplifting".
#    - **Mood fit:** clamped to 0–10. If it is missing, the score defaults to
#      3, so stories the model actually judged rank above it.
#    - **`is_uplifting`:** if missing, defaults to true, because every source
#      is a curated positive-news outlet.
#    Each default used is noted in the trace.
# 5. **Rank.** Non-uplifting stories are dropped, and the rest are sorted by
#    `final_score`, newest first on ties. The top `CLASSIFIER_TOP_N` are
#    returned with their scores, along with the rejected stories so the UI
#    can show what was filtered out and why.

# %%
DEFAULT_MOOD_FIT = 3


def _excerpt(article):
    return _truncate(" ".join(article.text.split()), CLASSIFIER_EXCERPT_CHARS)


def build_classifier_messages(intent, article):
    user = CLASSIFIER_USER_TEMPLATE.format(
        emotion=intent.emotion,
        desired_tone=intent.desired_tone,
        raw_mood=intent.raw_mood or "(no mood given)",
        title=article.title,
        source=article.source_name,
        excerpt=_excerpt(article),
    )
    return [{"role": "system", "content": CLASSIFIER_SYSTEM}, {"role": "user", "content": user}]


def _run_batches(batch_messages, log, max_new_tokens):
    outputs = []
    for i in range(0, len(batch_messages), CLASSIFIER_BATCH_SIZE):
        chunk = batch_messages[i : i + CLASSIFIER_BATCH_SIZE]
        try:
            results = llm.generate_batch("classifier", chunk, max_new_tokens=max_new_tokens, temperature=0.0)
            outputs.extend(r.text for r in results)
            log(f"Batch of {len(chunk)}: {sum(r.output_token_count for r in results)} tokens out "
                f"in {results[0].elapsed_seconds:.1f}s")
        except Exception as e:
            # One failed batch must not lose the whole request; its stories get default verdicts.
            log(f"Batch of {len(chunk)} failed ({type(e).__name__}: {str(e)[:120]}); using defaults.", fallback=True)
            outputs.extend([""] * len(chunk))
    return outputs


def _verdict(intent, article, raw, method):
    notes = []
    category = normalize_category(raw.get("category"))
    if category is None:
        hinted = categories_from_hints(article.title)
        category = hinted[0] if hinted else "Other Uplifting"
        notes.append(f"category {raw.get('category')!r} not allowed -> {category!r}")
    uplifting = to_bool(raw.get("is_uplifting"))
    if uplifting is None:
        uplifting = True
        notes.append("is_uplifting missing -> true (curated positive-news source)")
    score = to_score(raw.get("mood_fit_score"))
    if score is None:
        score = DEFAULT_MOOD_FIT
        notes.append(f"mood_fit_score missing -> {DEFAULT_MOOD_FIT}")
    reason = raw.get("one_line_reason")
    reason = reason.strip()[:200] if isinstance(reason, str) and reason.strip() else "No reason given."

    match = category in intent.target_categories
    final = score + (CATEGORY_MATCH_BONUS if match else 0.0) + 0.1 * min(article.prefilter_score, 5)
    story = ClassifiedStory(article, category, uplifting, score, reason, match, round(final, 2), method)
    return story, notes


def classify_stories(intent, articles, trace=None):
    trace = trace if trace is not None else []
    start = time.perf_counter()

    def log(message, fallback=False):
        trace.append(TraceEvent("classifier", message, round(time.perf_counter() - start, 2), fallback))

    if not articles:
        log("No articles to classify.")
        return []
    messages = [build_classifier_messages(intent, a) for a in articles]
    outputs = _run_batches(messages, log, max_new_tokens=120)

    parsed = [parse_json_object(o) for o in outputs]
    retry_idx = [i for i, p in enumerate(parsed) if p is None and outputs[i]]
    if retry_idx:
        log(f"{len(retry_idx)} repl{'y' if len(retry_idx) == 1 else 'ies'} not valid JSON; asking once more.")
        retry_messages = [
            messages[i] + [{"role": "assistant", "content": outputs[i]}, {"role": "user", "content": JSON_RETRY_PROMPT}]
            for i in retry_idx
        ]
        for i, text in zip(retry_idx, _run_batches(retry_messages, log, max_new_tokens=120)):
            if parse_json_object(text) is not None:
                outputs[i] = text

    stories = []
    for article, text in zip(articles, outputs):
        raw, method = parse_model_output(text, CLASSIFIER_FIELDS)
        story, notes = _verdict(intent, article, raw, method)
        stories.append(story)
        verdict = "uplifting" if story.is_uplifting else "NOT uplifting"
        log(f"'{article.title[:60]}': {story.category}, {verdict}, fit {story.mood_fit_score}/10, "
            f"final {story.final_score} [{method}]{' — ' + '; '.join(notes) if notes else ''}",
            fallback=bool(notes) or method != "json")
    return stories


def rank_stories(stories, top_n=CLASSIFIER_TOP_N, trace=None):
    rejected = [s for s in stories if not s.is_uplifting]
    kept = sorted(
        (s for s in stories if s.is_uplifting),
        key=lambda s: (-s.final_score, -(s.article.published.timestamp() if s.article.published else 0)),
    )
    if trace is not None:
        trace.append(TraceEvent(
            "classifier",
            f"Ranked {len(kept)} uplifting stories, rejected {len(rejected)}; returning top {min(top_n, len(kept))}."
            + (f" Only {len(kept)} passed (wanted {top_n}); not padding with rejected stories."
               if len(kept) < top_n else ""),
        ))
    return kept[:top_n], rejected

# %% [markdown]
# ### Test: parser robustness + three example moods
#
# **Parser check.** First, `parse_model_output` is fed the kinds of broken
# replies small models really produce: fenced, chatty, single-quoted,
# truncated, and pure garbage. Each line shows what was recovered and how.
# Then a garbage classifier reply goes through `_verdict` to show it ends in
# a valid, clearly marked default verdict rather than an exception.
#
# **Three moods end to end.** Then three moods each run through Mood →
# Retrieval → Classifier/Ranker. For each mood, the cell prints:
# - the parsed intent
# - the top stories, with category, mood-fit score, category-match flag,
#   final score and reason
# - any rejected stories
#
# In mock mode the model replies are canned, so the scores are arbitrary.
# Only the plumbing is being tested.

# %%
print("Parser robustness:")
for label, reply in [
    ("fenced", '```json\n{"category": "Animals & Nature", "is_uplifting": true, "mood_fit_score": 8, "one_line_reason": "Cute."}\n```'),
    ("chatty", 'Sure! Here is my answer: {"category": "animals", "is_uplifting": "yes", "mood_fit_score": "7/10", "one_line_reason": "Fits."} Hope that helps!'),
    ("python-ish", "{'category': 'Innovation', 'is_uplifting': True, 'mood_fit_score': 6,}"),
    ("truncated", '{"category": "Health & Well-being", "is_uplifting": false, "mood_fit_score": 2, "one_line_reason": "Mostly about a hosp'),
    ("garbage", "I think this story is nice."),
    ("empty", ""),
]:
    raw, method = parse_model_output(reply, CLASSIFIER_FIELDS)
    print(f"  {label:<11} -> {method:<7} {raw}")

_sample_article = (live_result.articles or fallback_result.articles)[0]
_story, _notes = _verdict(test_intent, _sample_article, {}, "default")
print(f"\nGarbage reply -> {_story.category}, uplifting={_story.is_uplifting}, fit={_story.mood_fit_score}; notes: {_notes}")

TEST_MOODS = [
    "stressed about exams",
    "feeling lonely since moving to a new city",
    "bored and want something fun, maybe about animals",
]
for mood in TEST_MOODS:
    trace = []
    intent = run_mood_agent(mood, trace)
    retrieval = retrieve_stories(intent, trace=trace)
    classified = classify_stories(intent, retrieval.articles, trace)
    top, rejected = rank_stories(classified, trace=trace)

    print("\n" + "=" * 110)
    print(f"MOOD: {mood!r}")
    print(f"  intent [{intent.parse_method}]: emotion={intent.emotion!r}, tone={intent.desired_tone!r}")
    print(f"  categories={intent.target_categories}  keywords={intent.keywords}")
    print(f"  retrieval: {len(retrieval.articles)} articles (fallback used: {retrieval.used_fallback})")
    print(f"\n  {'#':>2} {'final':>5} {'fit':>3} {'cat?':>4}  {'category':<26} title / reason")
    for i, s in enumerate(top, 1):
        print(f"  {i:>2} {s.final_score:>5.1f} {s.mood_fit_score:>3} {'yes' if s.category_match else '':>4}  "
              f"{s.category:<26} {s.article.title[:60]}")
        print(f"  {'':>45}{s.reason[:90]}  [{s.parse_method}]")
    for s in rejected:
        print(f"  REJECTED (not uplifting): {s.article.title[:70]} — {s.reason[:70]}")
    n_fallback = sum(ev.fallback_used for ev in trace)
    print(f"\n  trace: {len(trace)} events, {n_fallback} involving a fallback/default")

# %% [markdown]
# ## 13. Agent 5: Validator (plain Python)
#
# The Validator makes the output rules *guaranteed*, not just requested. It
# is deterministic Python with no model involved.
#
# - **Word count.** `count_words` counts whitespace-separated tokens that
#   contain at least one letter or digit, so a stray "—" or "…" never counts
#   as a word. The UI shows exactly this number.
# - **Cleaning first.** Before counting, `clean_writer_output` strips things
#   models add that aren't part of the text:
#   - lead-ins such as "Here is the summary:" or "Summary:"
#   - markdown headings and `**bold**` markers
#   - quotes wrapped around the whole reply
#   - trailing word-count notes such as "(82 words)"
#   - a plain title line on top: short, no closing punctuation, followed by
#     the body. A test model did this even though the prompt said no title.
# - **Retry loop.** The loop itself runs in the Writer, which owns the model.
#   If a summary is outside 70–90 words, the Writer is shown its own summary
#   plus feedback from `summary_feedback`. The feedback states the exact
#   count and how many words to add or remove, for example: "your summary has
#   64 words, but it must be between 70 and 90 words. Add about 16 words…".
#   This repeats up to `MAX_SUMMARY_RETRIES` (3) times.
# - **Last resort.** If all retries miss, `finalize_summary` takes the attempt
#   closest to the range and fixes it in code. The result is flagged in
#   `summary_status` and `validation_notes`, never passed off as a normal
#   result.
#   - **Too long:** cut at the last sentence boundary that leaves 70–90
#     words. If no boundary works, hard-cut at 90 words and end with "…".
#   - **Too short:** append whole sentences from the paraphrase (the model's
#     own words), then from the source if needed, until it reaches 70.
#     Sentences already present, or whose content words are mostly in the
#     summary already, are skipped so the result doesn't repeat itself. If that overshoots 90, it is
#     trimmed as above.
#   - If there is truly nothing to add, the summary is kept as is and marked
#     `out_of_range`.
# - **Copy check.** The paraphrase must be in the model's own words, so it is
#   compared with the source text, lowercased and as word tokens, in two ways:
#   - **Longest common n-gram:** the longest run of consecutive words that
#     appears in both, computed by dynamic programming. Short shared runs
#     are normal (names, "the city council"). A run of
#     `COPY_MAX_COMMON_NGRAM` (12) or more words is a copied sentence.
#   - **5-gram overlap:** the share of the paraphrase's 5-word sequences that
#     also appear in the source. Above `COPY_MAX_5GRAM_RATIO` (30%) means
#     the wording is largely lifted.
#
#   Either one flags the story (`copy_flagged`), and the note quotes the
#   shared run. The summary gets the same check, reported in the notes but not
#   flagged.
# - **Fact check.** The Writer is told to add nothing, and this checks it
#   without another model. From the paraphrase and the summary it pulls out
#   the facts a rewrite most often gets wrong, then looks for each one in the
#   source (the article text plus its title and source name):
#   - **Numbers,** including percentages, money and years. Formats are
#     normalized first, so "1,200" matches "1200", "50%" matches "50
#     percent", "1.5 million" matches "1,500,000", and a "2" in the rewrite
#     matches "two" in the source (spelled-out numbers up to twenty).
#   - **Names:** runs of 2 or more capitalized words, such as "Maria Lopez"
#     or "Riverside Food Bank". A name counts as found if each of its words
#     appears in the source, ignoring titles ("Dr.") and initials. An
#     ordinary capitalized word at the start of a sentence ("After Sam
#     Ortiz") is ignored.
#
#   Each text gets a count of facts checked and found, and a list of those
#   not found. A **warning** (`facts_flagged`) is raised for any missing
#   number, or for more than `FACT_MAX_UNSUPPORTED_NAMES` (2) missing names.
#   It is only a warning: the story is never blocked or regenerated, because
#   a false alarm would throw away a correct story. The checks are tuned to
#   miss some invented facts rather than flag correct ones.
#
#   **Known limits of the fact check:**
#   - Numbers written as words in the rewrite ("a dozen", "three hundred")
#     and rounded or derived numbers ("about a third" for 33%) are not
#     matched.
#   - Synonyms and changed descriptions ("a teacher" for "an educator") and
#     invented single-word names ("Lopez") are not checked at all.
#   - A real name the source spells differently, or gives only with a
#     title or nickname, can show up as not found.
#   - It checks that a fact appears in the source, not that it is used
#     correctly: a real name attached to the wrong deed passes.

# %%
def count_words(text):
    return sum(1 for token in (text or "").split() if re.search(r"[A-Za-z0-9]", token))


_LEADIN_RE = re.compile(
    r"^\s*(?:(?:sure[,!.]?\s*)?here(?:'s| is)[^:\n]{0,60}:|(?:rewritten story|rewrite|paraphrase|summary)\s*:)\s*",
    re.I,
)
_WORDCOUNT_TAIL_RE = re.compile(r"\s*[(\[]\s*(?:word count:?\s*)?\d+\s+words?\s*[)\]]\.?\s*$", re.I)


def clean_writer_output(text, single_paragraph=False):
    text = (text or "").strip()
    text = re.sub(r"^#+\s.*(?:\n|$)", "", text)  # a markdown heading line, e.g. a repeated title
    text = _LEADIN_RE.sub("", text)
    lines = text.split("\n", 1)
    # A plain title line on top (short, no closing punctuation, followed by the body); seen from real models.
    if len(lines) == 2 and lines[1].strip() and count_words(lines[0]) <= 15 and not lines[0].rstrip().endswith(
            (".", "!", "?", ":", '"', "”")):
        text = lines[1].strip()
    text = _WORDCOUNT_TAIL_RE.sub("", text).replace("**", "").strip()
    if len(text) > 1 and text[0] in "\"“" and text[-1] in "\"”":
        text = text[1:-1].strip()
    if single_paragraph:
        return " ".join(text.split())
    return "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip())


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])[\"”’']?\s+(?=[A-Z0-9\"“‘'(])", text or "") if s.strip()]


def summary_feedback(word_count):
    if word_count < SUMMARY_MIN_WORDS:
        direction = (f"Add about {SUMMARY_TARGET_WORDS - word_count} words of further detail from the article "
                     "(who was involved, what happened, why it matters).")
    else:
        direction = (f"Remove about {word_count - SUMMARY_TARGET_WORDS} words: cut minor details and keep the "
                     "main point.")
    return SUMMARY_FEEDBACK_TEMPLATE.format(
        count=word_count, min_words=SUMMARY_MIN_WORDS, max_words=SUMMARY_MAX_WORDS, direction=direction,
    )


def _hard_cut(text, max_words):
    kept, n = [], 0
    for token in text.split():
        is_word = bool(re.search(r"[A-Za-z0-9]", token))
        if is_word and n == max_words:
            break
        kept.append(token)
        n += is_word
    cut = " ".join(kept).rstrip(",;:—-– ")
    return cut if cut.endswith((".", "!", "?")) else cut + "…"


def _trim(text):
    kept = []
    for sentence in split_sentences(text):
        if count_words(" ".join(kept + [sentence])) > SUMMARY_MAX_WORDS:
            break
        kept.append(sentence)
    if count_words(" ".join(kept)) >= SUMMARY_MIN_WORDS:
        return " ".join(kept), "trimmed at a sentence boundary"
    return _hard_cut(text, SUMMARY_MAX_WORDS), f"hard-cut to {SUMMARY_MAX_WORDS} words"


def _extend(text, sources):
    present = {_norm_sentence(s) for s in split_sentences(text)}
    out, added, leftover = text.strip(), 0, None
    for source in sources:
        for sentence in split_sentences(source):
            if count_words(out) >= SUMMARY_MIN_WORDS:
                break
            if _norm_sentence(sentence) in present or _mostly_repeats(sentence, out):
                continue
            if count_words(f"{out} {sentence}") <= SUMMARY_MAX_WORDS:
                out, added = f"{out} {sentence}".strip(), added + 1
                present.add(_norm_sentence(sentence))
            elif leftover is None:
                leftover = sentence
    if count_words(out) < SUMMARY_MIN_WORDS and leftover:
        # No sentence fits whole; take the long one and cut it at the limit.
        out, added = _hard_cut(f"{out} {leftover}", SUMMARY_MAX_WORDS), added + 1
    return out, added


def _norm_sentence(sentence):
    return " ".join(re.findall(r"[a-z0-9]+", sentence.lower()))


_PRONOUNS = {"he", "she", "they", "them", "their", "his", "her", "its", "who", "which", "was", "were", "are"}


def _stems(text):
    # 4-letter prefixes are a crude stemmer, enough to match "helps"/"helped", "reunited"/"reunification".
    return {w[:4] for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 2 and w not in STOPWORDS and w not in _PRONOUNS}


def _mostly_repeats(sentence, text, threshold=0.6):
    # A paraphrase sentence often restates the summary's opening; appending it would read as a repeat.
    stems = _stems(sentence)
    return not stems or len(stems & _stems(text)) / len(stems) >= threshold


def finalize_summary(text, paraphrase, source_text):
    """Programmatic last resort. Returns (summary, status, note)."""
    n = count_words(text)
    if SUMMARY_MIN_WORDS <= n <= SUMMARY_MAX_WORDS:
        return text, "ok", None
    if n > SUMMARY_MAX_WORDS:
        fixed, how = _trim(text)
        return fixed, "trimmed", f"Summary {how} by the Validator after retries ({n} -> {count_words(fixed)} words)."
    fixed, added = _extend(text, [paraphrase, source_text])
    m = count_words(fixed)
    if SUMMARY_MIN_WORDS <= m <= SUMMARY_MAX_WORDS:
        return fixed, "extended", (f"Summary extended by the Validator with {added} sentence(s) from the "
                                   f"paraphrase/source after retries ({n} -> {m} words).")
    return fixed, "out_of_range", f"Summary still {m} words after retries and extension; shown as is."


def _word_tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", (text or "").lower())


def longest_common_ngram(a_tokens, b_tokens):
    """Longest run of consecutive tokens present in both lists (classic DP, one row at a time)."""
    best_len, best_end = 0, 0
    prev = [0] * (len(b_tokens) + 1)
    for i in range(1, len(a_tokens) + 1):
        cur = [0] * (len(b_tokens) + 1)
        a = a_tokens[i - 1]
        for j in range(1, len(b_tokens) + 1):
            if a == b_tokens[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, best_end = cur[j], i
        prev = cur
    return best_len, " ".join(a_tokens[best_end - best_len : best_end])


def ngram_overlap_ratio(a_tokens, b_tokens, n=5):
    a_grams = [tuple(a_tokens[i : i + n]) for i in range(len(a_tokens) - n + 1)]
    if not a_grams:
        return 0.0
    b_grams = {tuple(b_tokens[i : i + n]) for i in range(len(b_tokens) - n + 1)}
    return sum(g in b_grams for g in a_grams) / len(a_grams)


def copy_check(candidate, source_text):
    a, b = _word_tokens(candidate), _word_tokens(source_text)
    run, phrase = longest_common_ngram(a, b)
    ratio = ngram_overlap_ratio(a, b)
    flagged = run >= COPY_MAX_COMMON_NGRAM or ratio > COPY_MAX_5GRAM_RATIO
    return run, ratio, flagged, phrase


_SPELLED_NUMBERS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
# The lookbehind skips digits glued to letters or decimal points ("v2", "B12").
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
                        r"(?:\s*(thousand|million|billion)\b)?(?:\s*(?:%|percent\b|per cent\b))?", re.I)
_TITLES = {"dr", "mr", "mrs", "ms", "mx", "prof", "sir", "st", "rev", "jr", "sr"}
_NAME_JOINERS = {"of", "de", "da", "del", "van", "von", "la", "le"}


def _number_forms(match):
    """Every spelling a number may take in the source: '1,200' and '1200' -> '1200'; '1.2 million' -> '1.2', '1200000'."""
    whole, decimals, scale = match.group(1).replace(",", ""), (match.group(2) or "").rstrip("0"), match.group(3)
    value = f"{whole}.{decimals}" if decimals else whole
    forms = {value}
    if scale:
        forms.add(str(round(float(value) * _SCALES[scale.lower()])))
    return forms


def _source_numbers(reference):
    forms = set()
    for m in _NUMBER_RE.finditer(reference):
        forms |= _number_forms(m)
    # "Two volunteers" in the source must support "2 volunteers" in the rewrite.
    forms |= {_SPELLED_NUMBERS[w] for w in re.findall(r"[a-z]+", reference.lower()) if w in _SPELLED_NUMBERS}
    return forms


def _candidate_names(text, common_words):
    """Runs of 2+ capitalized words ('Maria Lopez', 'Dr. Amy Chen', 'Riverside Food Bank'), the usual shape of a
    name. Returns (name, starts_a_sentence) pairs."""
    names, run, sentence_start, run_at_start = [], [], True, False

    def close():
        while run and run[-1] in _NAME_JOINERS:
            run.pop()
        if sum(w[:1].isupper() for w in run) >= 2:
            names.append((" ".join(run), run_at_start))
        run.clear()

    for raw in (text or "").split():
        bare = raw.rstrip("\"”’')],;:")
        word = raw.strip("\"“”‘’'()[],;:!?").rstrip(".")
        is_title = word.lower() in _TITLES
        # "Dr." and initials like "J." end in a period without ending the sentence.
        ends_sentence = bare.endswith(("!", "?")) or (bare.endswith(".") and not is_title and len(word) > 1)
        # Every sentence starts with a capital, so "Today Maria Lopez" must lose "Today" but keep "Maria".
        common_start = sentence_start and (word.lower() in STOPWORDS | _PRONOUNS or word.lower() in common_words)
        if re.fullmatch(r"[A-Z][\w'’.\-]*", word) and not common_start:
            run_at_start = run_at_start if run else sentence_start
            run.append(word + "." if is_title and bare.endswith(".") else word)
        elif run and word in _NAME_JOINERS:
            run.append(word)
        else:
            close()
        if ends_sentence or raw.endswith((",", ";", ":")):
            close()
        sentence_start = ends_sentence
    close()
    return names


def _name_supported(name, reference_words):
    # Word by word, not as a phrase, so "Lopez, Maria" or a dropped middle name still counts. Titles and
    # one-letter pieces (initials, the "s" of "Lopez's") are skipped: missing them is not an invented fact.
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in _TITLES and w not in _NAME_JOINERS
             and len(w) > 1]
    return all(w in reference_words for w in words)


def fact_check(candidate, reference):
    """Names and numbers in candidate that the reference (title + source + article text) does not contain."""
    numbers = list(dict.fromkeys(m.group(0).strip() for m in _NUMBER_RE.finditer(candidate or "")))
    common_words = set(re.findall(r"\b[a-z][a-z'’-]*\b", f"{candidate} {reference}"))
    names = {}  # name -> whether its first use starts a sentence
    for name, at_start in _candidate_names(candidate, common_words):
        names.setdefault(name, at_start)
    source_numbers = _source_numbers(reference)
    reference_words = set(re.findall(r"[a-z0-9]+", reference.lower()))
    bad_numbers = [n for n in numbers if not _number_forms(_NUMBER_RE.match(n)) & source_numbers]
    # A sentence-initial run may begin with an ordinary word ("After Sam Ortiz"); retry without it before flagging.
    bad_names = [n for n, at_start in names.items() if not _name_supported(n, reference_words)
                 and not (at_start and _name_supported(n.split(" ", 1)[1], reference_words))]
    checked = len(numbers) + len(names)
    return {
        "checked": checked,
        "supported": checked - len(bad_numbers) - len(bad_names),
        "unsupported_numbers": bad_numbers,
        "unsupported_names": bad_names,
        # A wrong number changes the story's meaning; one or two odd names are often a missed match.
        "flagged": bool(bad_numbers) or len(bad_names) > FACT_MAX_UNSUPPORTED_NAMES,
    }

# %% [markdown]
# ## 14. Agent 4: Writer (large LLM)
#
# For each top-ranked story, the Writer produces:
# - **(a)** a shorter paraphrased version in its own words (120–200 words)
# - **(b)** a 70–90 word summary
#
# **Two focused calls, not one structured call.** With one call, the model
# would have to return both texts inside JSON. Long prose inside JSON is
# fragile for open models (quotes and newlines break it), and a summary retry
# would also regenerate the paraphrase. Separate plain-text calls avoid both
# problems, and each prompt can focus on one job:
#
# - **Paraphrase prompt:** reword and restructure everything, don't copy
#   sentences, keep names and numbers exact, add nothing, keep the reader's
#   desired tone.
# - **Summary prompt:** one paragraph of 70–90 words, aiming for 80, covering
#   who, what and why it is uplifting.
#
# Both share a system prompt that forbids invented facts and asks for plain
# prose with no titles, lists or word counts. Each prompt starts with a task
# marker (`TASK: PARAPHRASE` / `TASK: SUMMARY`), which the mock backend uses
# to choose its reply. It's harmless to the real model.
#
# **Batched across stories.** The Writer runs in rounds: all paraphrases in
# one batch, all first summaries in one batch, then up to 3 retry rounds
# containing only the summaries that missed 70–90 words. A retry shows the
# model its previous summary plus the Validator's exact-count feedback. With
# 5 stories that is 2 batched calls when everything passes, and at most 5.
# The source text is cut to `WRITER_SOURCE_CHARS` (~1k tokens) per prompt.
# Sampling is on (`temperature=0.7`) so the wording moves away from the
# source and retries produce different text.
#
# **Failures never crash.**
# - If a whole batch fails, its stories get empty outputs.
# - An empty summary goes through the same retry and Validator path.
#   As a last resort, it is assembled from paraphrase or source sentences
#   and flagged.
# - A missing paraphrase is noted and shown as unavailable.
#
# **Output:** one `StoryDetail` per story, with `validation_notes` and a
# `CallMetric` for every model call. Batched calls report the whole batch's
# time, because the rows run together.

# %%
WRITER_SYSTEM_PROMPT = """You are a careful writer for an uplifting-news app. You retell real news stories accurately in your own words.
Use only facts stated in the article you are given: never add names, numbers, quotes, places or events that are not in it.
Write plain prose: no title, no headings, no bullet points, no labels and no word counts."""

PARAPHRASE_USER_TEMPLATE = TASK_PARAPHRASE + """
Rewrite the article below as a shorter version of {min_words} to {max_words} words, entirely in your own words.
- Do not copy sentences or long phrases from the article: restructure and reword everything.
- Keep every name, number and fact accurate.
- Keep the tone {tone}.
Reply with only the rewritten story.

Title: {title}
Article:
{text}"""

SUMMARY_USER_TEMPLATE = TASK_SUMMARY + """
Summarize the article below in ONE paragraph of {min_words} to {max_words} words (aim for about {target_words} words).
Say who was involved, what happened, and why it is good news. Use your own words.
Reply with only the summary paragraph.

Title: {title}
Article:
{text}"""

SUMMARY_FEEDBACK_TEMPLATE = SUMMARY_FEEDBACK_MARKER + (
    " your summary has {count} words, but it must be between {min_words} and {max_words} words. {direction}"
    " Use only facts from the article. Reply with only the corrected summary paragraph."
)

PARAPHRASE_MAX_NEW_TOKENS = 360  # 200 words ≈ 270 tokens, plus headroom so a slightly long draft isn't cut off
SUMMARY_MAX_NEW_TOKENS = 180  # 90 words ≈ 120 tokens


def _writer_source(article):
    return _truncate(article.text.strip(), WRITER_SOURCE_CHARS)


def build_paraphrase_messages(intent, article):
    user = PARAPHRASE_USER_TEMPLATE.format(
        min_words=PARAPHRASE_MIN_WORDS, max_words=PARAPHRASE_MAX_WORDS, tone=intent.desired_tone,
        title=article.title, text=_writer_source(article),
    )
    return [{"role": "system", "content": WRITER_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_summary_messages(article):
    user = SUMMARY_USER_TEMPLATE.format(
        min_words=SUMMARY_MIN_WORDS, max_words=SUMMARY_MAX_WORDS, target_words=SUMMARY_TARGET_WORDS,
        title=article.title, text=_writer_source(article),
    )
    return [{"role": "system", "content": WRITER_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _metric(purpose, result):
    return CallMetric(purpose, result.input_token_count, result.output_token_count, round(result.elapsed_seconds, 2),
                      round(result.tokens_per_second, 1), result.peak_gpu_memory_gb, result.backend)


def _writer_batches(batch_messages, max_new_tokens, label, log):
    results = []
    for i in range(0, len(batch_messages), WRITER_BATCH_SIZE):
        chunk = batch_messages[i : i + WRITER_BATCH_SIZE]
        try:
            out = llm.generate_batch("writer", chunk, max_new_tokens=max_new_tokens, temperature=WRITER_TEMPERATURE)
            results.extend(out)
            log(f"{label}: batch of {len(chunk)}, {sum(r.output_token_count for r in out)} tokens out "
                f"in {out[0].elapsed_seconds:.1f}s")
        except Exception as e:
            # A failed batch leaves its stories with empty text, which the Validator path then handles.
            log(f"{label}: batch of {len(chunk)} failed ({type(e).__name__}: {str(e)[:120]}).", fallback=True)
            results.extend([None] * len(chunk))
    return results


def write_stories(stories, intent, trace=None):
    trace = trace if trace is not None else []
    start = time.perf_counter()

    def log(message, fallback=False):
        trace.append(TraceEvent("writer", message, round(time.perf_counter() - start, 2), fallback))

    if not stories:
        log("No stories to write.")
        return []
    articles = [s.article for s in stories]
    calls = [[] for _ in stories]

    # (a) Paraphrases: one batched round.
    paraphrases = []
    for i, r in enumerate(_writer_batches([build_paraphrase_messages(intent, a) for a in articles],
                                          PARAPHRASE_MAX_NEW_TOKENS, "paraphrase", log)):
        if r is not None:
            calls[i].append(_metric("paraphrase", r))
        paraphrases.append(clean_writer_output(r.text) if r else "")

    # (b) Summaries: first attempt for all, then retry rounds with exact-count feedback for the misses.
    base = [build_summary_messages(a) for a in articles]
    conversations = list(base)
    attempts = [[] for _ in stories]
    pending = list(range(len(stories)))
    for attempt_no in range(1, MAX_SUMMARY_RETRIES + 2):
        if not pending:
            break
        label = "summary" if attempt_no == 1 else f"summary retry {attempt_no - 1}"
        results = _writer_batches([conversations[i] for i in pending], SUMMARY_MAX_NEW_TOKENS, label, log)
        still_pending = []
        for i, r in zip(pending, results):
            if r is not None:
                calls[i].append(_metric(f"summary attempt {attempt_no}", r))
            text = clean_writer_output(r.text, single_paragraph=True) if r else ""
            n = count_words(text)
            attempts[i].append((text, n))
            if SUMMARY_MIN_WORDS <= n <= SUMMARY_MAX_WORDS:
                continue
            if attempt_no <= MAX_SUMMARY_RETRIES:
                # Only the latest attempt is shown back, so the prompt doesn't grow with each retry.
                conversations[i] = base[i] + [
                    {"role": "assistant", "content": text or "(no summary)"},
                    {"role": "user", "content": summary_feedback(n)},
                ]
                still_pending.append(i)
        if still_pending:
            # Not a fallback: a retry is the normal repair path (counted under metrics "retries").
            log(f"{len(still_pending)} summar{'y' if len(still_pending) == 1 else 'ies'} outside "
                f"{SUMMARY_MIN_WORDS}-{SUMMARY_MAX_WORDS} words after attempt {attempt_no}: "
                f"{[attempts[i][-1][1] for i in still_pending]} words; retrying.")
        pending = still_pending

    # Final validation (last-resort fixes, copy checks) is the Validator's own work, timed and traced separately.
    v_start = time.perf_counter()

    def vlog(message, fallback=False):
        trace.append(TraceEvent("validator", message, round(time.perf_counter() - v_start, 3), fallback))

    details = []
    for i, story in enumerate(stories):
        article, paraphrase, notes = story.article, paraphrases[i], []
        word_counts = [n for _, n in attempts[i]]
        passing = [t for t, n in attempts[i] if SUMMARY_MIN_WORDS <= n <= SUMMARY_MAX_WORDS]
        if passing:
            summary = passing[0]
            status = "ok" if len(word_counts) == 1 else "ok_after_retry"
            if status == "ok_after_retry":
                notes.append(f"Summary passed after {len(word_counts) - 1} retr{'y' if len(word_counts) == 2 else 'ies'} "
                             f"(attempt word counts: {word_counts}).")
        else:
            def distance(n):
                return SUMMARY_MIN_WORDS - n if n < SUMMARY_MIN_WORDS else n - SUMMARY_MAX_WORDS
            closest = min(attempts[i], key=lambda tn: distance(tn[1]))[0]
            summary, status, note = finalize_summary(closest, paraphrase, article.text)
            notes.append(f"Summary missed {SUMMARY_MIN_WORDS}-{SUMMARY_MAX_WORDS} words on all "
                         f"{len(word_counts)} attempts {word_counts}. {note}")
        word_count = count_words(summary)

        if not paraphrase:
            notes.append("Writer produced no paraphrase for this story.")
        p_words = count_words(paraphrase)
        if paraphrase and not PARAPHRASE_MIN_WORDS <= p_words <= PARAPHRASE_MAX_WORDS:
            notes.append(f"Paraphrase is {p_words} words (asked for {PARAPHRASE_MIN_WORDS}-{PARAPHRASE_MAX_WORDS}).")

        run, ratio, flagged, phrase = copy_check(paraphrase, article.text) if paraphrase else (0, 0.0, False, "")
        if flagged:
            notes.append(f"COPY FLAG: paraphrase shares a {run}-word run with the source (\"{phrase[:120]}\") "
                         f"and {ratio:.0%} of its 5-word sequences.")
        elif paraphrase:
            notes.append(f"Copy check OK: longest shared run {run} words, 5-gram overlap {ratio:.0%}.")
        s_run, s_ratio, s_flagged, _ = copy_check(summary, article.text)
        if s_flagged:
            notes.append(f"Note: summary shares a {s_run}-word run with the source ({s_ratio:.0%} 5-gram overlap).")
        # Title and source name count as source text: a rewrite may name the outlet or repeat the headline.
        reference = f"{article.title}\n{article.source_name}\n{article.text}"
        p_facts, s_facts = fact_check(paraphrase, reference), fact_check(summary, reference)
        for label, facts in (("paraphrase", p_facts), ("summary", s_facts)):
            missing = facts["unsupported_numbers"] + facts["unsupported_names"]
            if missing:
                notes.append(f"{'FACT CHECK WARNING' if facts['flagged'] else 'Fact check'}: {len(missing)} of "
                             f"{facts['checked']} names/numbers in the {label} not found in the source: "
                             + ", ".join(f"'{m}'" for m in missing) + ".")
            elif facts["checked"]:
                notes.append(f"Fact check OK: all {facts['checked']} names/numbers in the {label} appear in the source.")
        facts_flagged = p_facts["flagged"] or s_facts["flagged"]
        if article.text_source == "fallback":
            notes.append("Source is a built-in placeholder story (offline fallback), not real news.")

        details.append(StoryDetail(
            title=article.title, source=article.source_name, url=article.url, published=article.published,
            category=story.category, mood_fit_score=story.mood_fit_score, final_score=story.final_score,
            paraphrase=paraphrase, paraphrase_word_count=p_words,
            summary=summary, word_count=word_count,
            summary_passed=SUMMARY_MIN_WORDS <= word_count <= SUMMARY_MAX_WORDS, summary_status=status,
            summary_attempt_word_counts=word_counts,
            copy_longest_ngram=run, copy_5gram_ratio=round(ratio, 3), copy_flagged=flagged,
            paraphrase_facts=p_facts, summary_facts=s_facts, facts_flagged=facts_flagged,
            is_fallback_story=article.text_source == "fallback",
            validation_notes=notes, calls=calls[i],
        ))
        vlog(f"'{article.title[:60]}': summary {word_count} words [{status}], paraphrase {p_words} words, "
             f"copy run {run}{' FLAGGED' if flagged else ''}, facts {p_facts['supported'] + s_facts['supported']}/"
             f"{p_facts['checked'] + s_facts['checked']} found{' WARNING' if facts_flagged else ''}.",
             fallback=status not in ("ok", "ok_after_retry") or flagged or facts_flagged)
    return details

# %% [markdown]
# ### Test: Validator checks + Writer end to end
#
# **Validator checks.** These use fixed inputs, so they behave the same in
# mock and real mode:
# - word counting ignores stray punctuation
# - cleaning removes a "Summary:" lead-in and a "(82 words)" tail
# - a 120-word summary is trimmed into range, and a 40-word one is extended
#   into range
# - a "paraphrase" pasted from the source text is flagged by the copy check,
#   while genuinely reworded text passes
# - fact check, on a small made-up source: (a) a clean paraphrase passes,
#   (b) an invented number raises a warning, (c) one invented name is listed
#   without a warning, while three raise one, (d) number-format variants
#   ("1200", "50%", "2" for "Two", "$15000", "1,500,000" for "1.5 million")
#   all pass, and (e) an ordinary word starting a sentence is not taken as
#   part of a name. Each case prints PASS or FAIL
#
# **End to end.** One mood runs through Mood → Retrieval → Classifier/Ranker
# → Writer, and every `StoryDetail` is printed:
# - paraphrase and summary, with word counts and status
# - validation notes
# - per-call metrics
#
# In mock mode the first summary attempt is deliberately 60 words, so every
# story should show `ok_after_retry` with attempt counts `[60, 80]`. The
# paraphrase is canned text, so its copy check passes trivially.

# %%
print("Validator checks:")
print("  count_words('Hello — world … 3 dogs!') =", count_words("Hello — world … 3 dogs!"), "(expect 4)")
print("  clean:", repr(clean_writer_output("Summary: A turtle went home. (82 words)", single_paragraph=True)))
print("  clean title line:", repr(clean_writer_output("Turtles Return to Ocean After Months of Care\n\nThe turtles swam home.")))

# Real 46-word reply from a test model whose paraphrase restates its first sentence: the restatement must be skipped.
_bus_summary = ("A city bus driver helped a lost child find her way home by staying with her for nearly 20 minutes, "
                "ensuring her safety and reunification with her grandmother. This act showcases community support "
                "during challenging situations, emphasizing the value of patience and empathy in such moments.")
_bus_para = ("A bus driver helps a lost child find their way home by staying with them during a moment of distress, "
             "ensuring they're taken care of and reunited with their grandmother.")
_fixed, _status, _note = finalize_summary(_bus_summary, _bus_para, FALLBACK_STORIES[7]["text"])
print(f"  extend skips restatement: {count_words(_bus_summary)} -> {count_words(_fixed)} words [{_status}], "
      f"restatement added: {'staying with them during a moment of distress' in _fixed} (expect False)")

_long = " ".join(f"Sentence number {i} tells part of the story about the volunteers and the turtles." for i in range(9))
_fixed, _status, _note = finalize_summary(_long, "", "")
print(f"  too long ({count_words(_long)} words) -> {count_words(_fixed)} words [{_status}] {_note}")

_src = FALLBACK_STORIES[1]["text"]
_short = "Volunteers released rescued sea turtles back into the ocean after months of care at a coastal center."
_fixed, _status, _note = finalize_summary(_short, "", _src)
print(f"  too short ({count_words(_short)} words) -> {count_words(_fixed)} words [{_status}]")

_copied = " ".join(_src.split()[:80])
print("  copy check, pasted text   (run, ratio, flagged):", copy_check(_copied, _src)[:3])
_reworded = ("After a bitter winter left several turtles too cold to swim, a small rescue team nursed them back to "
             "strength in heated pools and then watched them return to the sea, cheered on by local children.")
print("  copy check, reworded text (run, ratio, flagged):", copy_check(_reworded, _src)[:3])

# A made-up source (not a real article) with the kinds of facts a rewrite tends to alter.
_fact_src = ("Maria Lopez, 34, and the Riverside Food Bank delivered 1,200 meals in March 2024, a 50 percent rise on "
             "last year. Two volunteers, Dr. Amy Chen and Sam Ortiz, drove the vans through 12 towns. The bank raised "
             "$15,000 and hopes to reach 1.5 million people by 2030.")
_fact_cases = [
    ("clean paraphrase", "In March 2024, Maria Lopez and the Riverside Food Bank handed out 1,200 meals, up 50 percent. "
                         "Volunteers Amy Chen and Sam Ortiz drove the vans through 12 towns.", [], False),
    ("invented number", "The Riverside Food Bank handed out 1,500 meals in 2024.", ["1,500"], True),
    ("invented name", "Maria Lopez got help from Dr. Lee and the Riverside Food Bank.", ["Dr. Lee"], False),
    ("3 invented names", "Maria Lopez worked with Dr. Lee, John Park and Ana Silva.",
     ["Dr. Lee", "John Park", "Ana Silva"], True),
    ("number formats", "Lopez's team served 1200 meals, a 50% rise. 2 volunteers covered 12 towns, raised $15000 and "
                       "aim for 1,500,000 people by 2030.", [], False),
    ("sentence starts", "After Sam Ortiz joined, the Riverside Food Bank grew. Thanks to Dr. Amy Chen, it works.", [], False),
]
for _label, _text, _want_missing, _want_flag in _fact_cases:
    _r = fact_check(_text, _fact_src)
    _missing = _r["unsupported_numbers"] + _r["unsupported_names"]
    _ok = _missing == _want_missing and _r["flagged"] == _want_flag
    print(f"  fact check, {_label:<17} {_r['supported']}/{_r['checked']} found, missing {_missing}, "
          f"warning {_r['flagged']} -> {'PASS' if _ok else 'FAIL'}")

writer_mood = "bored and want something fun, maybe about animals"
writer_trace = []
_t0 = time.perf_counter()
writer_intent = run_mood_agent(writer_mood, writer_trace)
writer_retrieval = retrieve_stories(writer_intent, trace=writer_trace)
writer_top, _ = rank_stories(classify_stories(writer_intent, writer_retrieval.articles, writer_trace), trace=writer_trace)
story_details = write_stories(writer_top, writer_intent, writer_trace)
print(f"\nEnd to end for {writer_mood!r}: {len(story_details)} stories in {time.perf_counter() - _t0:.1f}s")

for i, d in enumerate(story_details, 1):
    print("\n" + "=" * 110)
    print(f"{i}. {d.title}\n   {d.source} | {d.category} | fit {d.mood_fit_score}/10 | final {d.final_score}\n   {d.url}")
    print(f"\n   PARAPHRASE ({d.paraphrase_word_count} words):\n   {d.paraphrase[:600]}{'…' if len(d.paraphrase) > 600 else ''}")
    print(f"\n   SUMMARY ({d.word_count} words, {d.summary_status}, attempts {d.summary_attempt_word_counts}):\n   {d.summary}")
    print("\n   Notes:")
    for note in d.validation_notes:
        print(f"   - {note}")
    print(f"\n   {'call':<20} {'in':>5} {'out':>5} {'secs':>6} {'tok/s':>6} {'peak GB':>8}")
    for c in d.calls:
        peak = f"{c.peak_gpu_memory_gb:.2f}" if c.peak_gpu_memory_gb is not None else "n/a"
        print(f"   {c.purpose:<20} {c.input_tokens:>5} {c.output_tokens:>5} {c.elapsed_seconds:>6.1f} "
              f"{c.tokens_per_second:>6.1f} {peak:>8}")

# %% [markdown]
# ## 15. Orchestrator
#
# `StoryOrchestrator` is the single entry point the UI uses. It has two steps.
#
# **1. `recommend(mood, n=5, preferred_source=None)`**, which runs:
#
# Mood/Intent → Retrieval → Classifier/Ranker
#
# It returns the top `n` (default 5) recommendations. Each has a
# `story_id`, title, source, URL, category, mood-fit and final scores, and the
# classifier's one-line reason. It also returns the rejected stories, the
# feed statuses, and any notes (fallback used, fewer than `n` stories
# passed). Nothing is written yet, so this step stays fast.
#
# If fewer than `n` live stories pass the uplifting check, it tops up with
# the built-in placeholder stories. These are classified the same way,
# tagged "placeholder", and flagged in the notes, so the UI always shows at
# least 5 stories. Rejected live stories are never shown as recommendations.
#
# **2. `get_story_details(story_id)`** runs the Writer + Validator **lazily**,
# only for the story the user actually opens.
# - The result is cached by `story_id`, a hash of the article URL, so
#   opening it again costs nothing and is reported as a cache hit.
# - The paraphrase uses the tone of the mood that first produced the story.
#   A cached story keeps that tone.
# - An unknown id returns an `error` field instead of raising.
#
# **Agent trace.** Every response includes `trace`, an ordered list of steps:
# - agent and model, where Retrieval and the Validator are plain Python
#   and labeled as such
# - status:
#   - `ok`
#   - `fallback`: the agent used a default or fallback somewhere
#   - `error`: the agent raised, and the orchestrator substituted a safe
#     default
#   - `cache_hit`
#   - `skipped`: the Validator, when the Writer errored
# - duration and a short output preview
# - the agent's detailed events
#
# The Writer and Validator run inside one function, but the Validator's
# final checks are timed separately, so they get their own step.
#
# **Metrics.** Aggregated per agent and in total:
# - calls, input and output tokens (from the real tokenizer)
# - wall time
# - tokens per second: output tokens ÷ the agent's wall time. Batched rows
#   share one `generate()`, so summing per-row rates would overcount.
# - peak GPU memory, plus currently allocated memory in the total
# - retries: classifier JSON repairs and summary re-prompts
# - cache hits: feeds, articles and story details
#
# Token counts come from `llm.call_log`, which records every generation the
# step made.
#
# **JSON-safe.** Every response is plain dicts, lists, strings and numbers,
# with dates as ISO strings, via `to_jsonable`. A broken agent (an
# unexpected exception) degrades the answer and is marked `error` in the
# trace. It never fails the request.
#
# **Live progress.** Both methods accept `on_step`, a callback that gets
# each finished step as soon as it completes. The UI uses it to update the
# Agent Trace live.
#
# The orchestrator holds its own LLM backend and switches the agents to it
# for the duration of a call (`use_backend`). That's what lets the mock test
# below run even on a GPU runtime where the global `llm` is the real
# ModelManager.

# %%
import hashlib
import math
import uuid
from contextlib import contextmanager
from dataclasses import field, fields, is_dataclass


AGENT_LABELS = {"mood": "Mood/Intent", "retrieval": "Retrieval", "classifier": "Classifier/Ranker",
                "writer": "Writer", "validator": "Validator"}
SEARCH_AGENTS = ["mood", "retrieval", "classifier"]  # recommend() steps, in order
DETAIL_AGENTS = ["writer", "validator"]  # get_story_details() steps, in order


@dataclass
class TraceStep:
    agent: str
    model: str
    status: str  # "ok" | "fallback" | "error" | "cache_hit" | "skipped"
    duration_s: float
    output_preview: str
    events: list[TraceEvent]


@dataclass
class AgentTrace:
    steps: list[TraceStep] = field(default_factory=list)


def to_jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None  # NaN/inf are not valid JSON
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


@contextmanager
def use_backend(backend):
    global llm
    previous, llm = llm, backend
    try:
        yield
    finally:
        llm = previous


def _clip(text, limit=160):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _empty_metrics():
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "wall_time_s": 0.0, "tokens_per_second": 0.0,
            "peak_gpu_memory_gb": None, "retries": 0, "cache_hits": 0}


class _RunRecorder:
    """Collects the AgentTrace and per-agent metrics for one orchestrator request."""

    def __init__(self, backend, on_step=None):
        self.backend = backend
        self.on_step = on_step  # called with each finished step (JSON-ready), e.g. to stream the trace to the UI
        self.trace = AgentTrace()
        self.metrics = {}
        # Mock replies take ~0 s, so a tokens/sec figure would be meaningless (and huge); report 0 instead.
        self._rates_meaningful = not isinstance(backend, MockModelManager)

    def _rate(self, tokens, seconds):
        return _rate(tokens, seconds, self._rates_meaningful)

    def _add_metrics(self, agent, calls, duration, retries=0, cache_hits=0):
        m = self.metrics.setdefault(agent, _empty_metrics())
        m["calls"] += len(calls)
        m["input_tokens"] += sum(c.input_token_count for c in calls)
        m["output_tokens"] += sum(c.output_token_count for c in calls)
        m["wall_time_s"] = round(m["wall_time_s"] + duration, 3)
        peaks = [c.peak_gpu_memory_gb for c in calls if c.peak_gpu_memory_gb is not None]
        if m["peak_gpu_memory_gb"] is not None:
            peaks.append(m["peak_gpu_memory_gb"])
        m["peak_gpu_memory_gb"] = round(max(peaks), 2) if peaks else None
        m["retries"] += retries
        m["cache_hits"] += cache_hits
        m["tokens_per_second"] = self._rate(m["output_tokens"], m["wall_time_s"])

    def add_step(self, agent, model, status, duration, preview, events, calls=(), retries=0, cache_hits=0):
        step = TraceStep(agent, model, status, round(duration, 3), _clip(preview), list(events))
        self.trace.steps.append(step)
        self._add_metrics(agent, list(calls), duration, retries, cache_hits)
        if self.on_step:
            try:
                self.on_step(to_jsonable(step))
            except Exception:
                pass  # a broken progress listener must never break the pipeline itself

    def run(self, agent, model, fn, on_error, preview, retries=None, cache_hits=None, split=None):
        """Run one agent step; split=(agent, model, preview_fn) carves that agent's events into their own step."""
        events, mark, start = [], len(self.backend.call_log), time.perf_counter()
        error = None
        try:
            result = fn(events)
        except Exception as e:
            # One broken agent should degrade the answer (safe default + 'error' in the trace), not fail the request.
            error = e
            events.append(TraceEvent(agent, f"ERROR {type(e).__name__}: {str(e)[:200]}; using a safe default.",
                                     round(time.perf_counter() - start, 3), True))
            result = on_error()
        duration = time.perf_counter() - start
        calls = self.backend.call_log[mark:]

        split_events = [e for e in events if split and e.agent == split[0]]
        own_events = [e for e in events if not (split and e.agent == split[0])]
        split_duration = max((e.elapsed_s for e in split_events), default=0.0)

        def status_of(evs):
            return "error" if error else ("fallback" if any(e.fallback_used for e in evs) else "ok")

        self.add_step(agent, model, status_of(own_events), duration - split_duration,
                      preview(result) if not error else f"error: {type(error).__name__}", own_events, calls,
                      retries(result, calls) if retries and not error else 0,
                      cache_hits(result) if cache_hits and not error else 0)
        if split and (split_events or error):
            self.add_step(split[0], split[1], "skipped" if error else status_of(split_events), split_duration,
                          split[2](result) if not error else "skipped (writer error)", split_events)
        return result

    def summary(self):
        return {"per_agent": self.metrics, "total": aggregate_totals(self.metrics, self._rates_meaningful)}


def _rate(tokens, seconds, meaningful=True):
    return round(tokens / seconds, 1) if meaningful and seconds > 0 else 0.0


def aggregate_totals(per_agent, rates_meaningful=True):
    total = _empty_metrics()
    for m in per_agent.values():
        for key in ("calls", "input_tokens", "output_tokens", "retries", "cache_hits"):
            total[key] += m[key]
        total["wall_time_s"] = round(total["wall_time_s"] + m["wall_time_s"], 3)
    peaks = [m["peak_gpu_memory_gb"] for m in per_agent.values() if m["peak_gpu_memory_gb"] is not None]
    total["peak_gpu_memory_gb"] = max(peaks) if peaks else None
    llm_wall = sum(m["wall_time_s"] for m in per_agent.values() if m["output_tokens"])
    total["tokens_per_second"] = _rate(total["output_tokens"], llm_wall, rates_meaningful)
    total["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2) if HAS_CUDA else None
    return total


def story_id_for(url):
    return hashlib.sha1(url.encode()).hexdigest()[:12]


class StoryOrchestrator:
    def __init__(self, backend):
        self.backend = backend
        self._stories = {}  # story_id -> (ClassifiedStory, MoodIntent) from the latest recommend() that returned it
        self._details = {}  # story_id -> JSON-ready StoryDetail (lazy cache)

    @property
    def is_mock(self):
        return isinstance(self.backend, MockModelManager)

    def _model(self, key):
        model_id = self.backend.model_ids[key]
        return f"mock of {model_id}" if self.is_mock else model_id

    def recommend(self, mood, n=CLASSIFIER_TOP_N, preferred_source=None, on_step=None):
        rec = _RunRecorder(self.backend, on_step)
        n = max(1, min(int(n or CLASSIFIER_TOP_N), PREFILTER_K))
        with use_backend(self.backend):
            intent = rec.run(
                "mood", self._model("mood"),
                fn=lambda ev: run_mood_agent(mood, ev),
                on_error=lambda: MoodIntent(mood or "", "unspecified", "warm and hopeful",
                                            categories_from_hints(mood or "")[:3],
                                            _clean_keywords([], mood or "", []), "default"),
                preview=lambda i: f"{i.emotion} → {i.desired_tone}; categories {i.target_categories}; "
                                  f"keywords {i.keywords}",
            )

            article_hits_before = ARTICLE_CACHE_STATS["hits"]
            retrieval = rec.run(
                "retrieval", "none (Python tool: RSS + trafilatura)",
                fn=lambda ev: retrieve_stories(intent, preferred_source=preferred_source, trace=ev),
                on_error=lambda: RetrievalResult(_fallback_articles(intent.keywords, PREFILTER_K), True, [], None),
                preview=lambda r: (f"{len(r.articles)} articles "
                                   f"({sum(a.text_source != 'fallback' for a in r.articles)} live, "
                                   f"{sum(a.text_source == 'fallback' for a in r.articles)} fallback); "
                                   f"feeds OK {sum(s.ok for s in r.feed_statuses)}/{len(r.feed_statuses)}"),
                cache_hits=lambda r: (sum(s.from_cache for s in r.feed_statuses)
                                      + ARTICLE_CACHE_STATS["hits"] - article_hits_before),
            )

            classified_count = [len(retrieval.articles)]

            def classify_and_rank(ev):
                top, rejected = rank_stories(classify_stories(intent, retrieval.articles, ev), top_n=n, trace=ev)
                if len(top) < n:
                    # The spec promises at least n recommendations; when too many live stories fail the
                    # uplifting check, fill the gap with the labeled built-in stories, judged the same way.
                    present = {s.article.url for s in top} | {a.url for a in retrieval.articles}
                    extra = [a for a in _fallback_articles(intent.keywords, len(FALLBACK_STORIES))
                             if a.url not in present][: n - len(top)]
                    ev.append(TraceEvent("classifier", f"Only {len(top)} stories passed the uplifting check; "
                                         f"topping up with {len(extra)} built-in placeholder stories.", 0.0, True))
                    classified_count[0] += len(extra)
                    top += rank_stories(classify_stories(intent, extra, ev), top_n=n - len(top))[0]
                return top, rejected

            top, rejected = rec.run(
                "classifier", self._model("classifier"),
                fn=classify_and_rank,
                on_error=lambda: rank_stories([_verdict(intent, a, {}, "default")[0] for a in retrieval.articles],
                                              top_n=n),
                preview=lambda tr: (f"top {len(tr[0])}: "
                                    + "; ".join(f"{s.article.title[:35]} ({s.final_score})" for s in tr[0][:3])
                                    + f"; rejected {len(tr[1])}"),
                # Calls beyond one per classified article are the JSON repair retries.
                retries=lambda tr, calls: max(0, len(calls) - classified_count[0]),
            )
        used_fallback = retrieval.used_fallback or any(s.article.text_source == "fallback" for s in top)

        recommendations = []
        for rank, story in enumerate(top, 1):
            sid = story_id_for(story.article.url)
            self._stories[sid] = (story, intent)
            a = story.article
            recommendations.append({
                "story_id": sid, "rank": rank, "title": a.title, "source": a.source_name, "url": a.url,
                "published": a.published, "category": story.category, "mood_fit_score": story.mood_fit_score,
                "final_score": story.final_score, "category_match": story.category_match, "reason": story.reason,
                "is_fallback_story": a.text_source == "fallback", "details_cached": sid in self._details,
            })

        notes = []
        if len(top) < n:
            notes.append(f"Only {len(top)} stories passed the uplifting check (asked for {n}).")
        if used_fallback:
            notes.append("Too few usable live stories this time; some results are built-in placeholder "
                         "stories, not real news.")
        if self.is_mock:
            notes.append("MOCK MODE: model outputs are canned placeholders, not real model output.")

        return to_jsonable({
            "run_id": uuid.uuid4().hex[:8],
            "mood": mood,
            "mock_llm": self.is_mock,
            "intent": intent,
            "recommendations": recommendations,
            "rejected": [{"title": s.article.title, "source": s.article.source_name, "reason": s.reason}
                         for s in rejected],
            "used_fallback": used_fallback,
            "preferred_source": retrieval.preferred_source,
            "retrieval_stats": {"urls_attempted": retrieval.urls_attempted, "urls_resolved": retrieval.urls_resolved,
                                "articles_live": sum(a.text_source != "fallback" for a in retrieval.articles),
                                "articles_fallback": sum(a.text_source == "fallback" for a in retrieval.articles)},
            "feeds": [{"source": s.source_name, "ok": s.ok, "entries": s.n_entries, "cached": s.from_cache,
                       "error": s.error} for s in retrieval.feed_statuses],
            "notes": notes,
            "trace": rec.trace.steps,
            "metrics": rec.summary(),
        })

    def get_story_details(self, story_id, on_step=None):
        rec = _RunRecorder(self.backend, on_step)
        if story_id in self._details:
            start = time.perf_counter()
            story = self._details[story_id]
            rec.add_step("writer", self._model("writer"), "cache_hit", time.perf_counter() - start,
                         f"cached: summary {story['word_count']} words [{story['summary_status']}]", [], cache_hits=1)
            return to_jsonable({"story_id": story_id, "cache_hit": True, "story": story,
                                "trace": rec.trace.steps, "metrics": rec.summary()})

        entry = self._stories.get(story_id)
        if entry is None:
            return {"story_id": story_id, "cache_hit": False, "story": None,
                    "error": "Unknown story_id: call recommend() first and use an id from its results.",
                    "trace": [], "metrics": rec.summary()}

        story, intent = entry
        with use_backend(self.backend):
            details = rec.run(
                "writer", self._model("writer"),
                fn=lambda ev: write_stories([story], intent, ev),
                on_error=lambda: [],
                preview=lambda ds: (f"paraphrase {ds[0].paraphrase_word_count} words; summary attempts "
                                    f"{ds[0].summary_attempt_word_counts}") if ds else "no output",
                retries=lambda ds, calls: sum(len(d.summary_attempt_word_counts) - 1 for d in ds),
                split=("validator", "none (Python rules)",
                       lambda ds: (f"summary {ds[0].word_count} words [{ds[0].summary_status}]; copy run "
                                   f"{ds[0].copy_longest_ngram}{' FLAGGED' if ds[0].copy_flagged else ''}; "
                                   f"facts {'WARNING' if ds[0].facts_flagged else 'ok'}")
                       if ds else "no output"),
            )
        if not details:
            return to_jsonable({"story_id": story_id, "cache_hit": False, "story": None,
                                "error": "The Writer failed for this story; see the trace.",
                                "trace": rec.trace.steps, "metrics": rec.summary()})

        story_json = to_jsonable(details[0])
        story_json["story_id"] = story_id
        self._details[story_id] = story_json
        return to_jsonable({"story_id": story_id, "cache_hit": False, "story": story_json,
                            "trace": rec.trace.steps, "metrics": rec.summary()})


def print_trace(steps):
    print(f"   {'agent':<11} {'model':<38} {'status':<9} {'secs':>7}  preview")
    for s in steps:
        print(f"   {s['agent']:<11} {s['model'][:38]:<38} {s['status']:<9} {s['duration_s']:>7.2f}  "
              f"{s['output_preview'][:90]}")


def print_metrics(metrics):
    print(f"   {'agent':<11} {'calls':>5} {'in tok':>7} {'out tok':>7} {'wall s':>7} {'tok/s':>6} "
          f"{'peak GB':>7} {'retries':>7} {'cache':>5}")
    for name, m in [*metrics["per_agent"].items(), ("TOTAL", metrics["total"])]:
        peak = f"{m['peak_gpu_memory_gb']:.2f}" if m["peak_gpu_memory_gb"] is not None else "n/a"
        print(f"   {name:<11} {m['calls']:>5} {m['input_tokens']:>7} {m['output_tokens']:>7} {m['wall_time_s']:>7.2f} "
              f"{m['tokens_per_second']:>6.1f} {peak:>7} {m['retries']:>7} {m['cache_hits']:>5}")


def run_orchestrator_demo(orch, mood, label):
    checks = []
    print(f"{label}: recommend({mood!r})")
    response = orch.recommend(mood, n=5)
    checks.append(("recommend() response is JSON-serializable", bool(json.dumps(response))))
    checks.append(("at least 5 recommendations", len(response["recommendations"]) >= 5))
    for note in response["notes"]:
        print(f"   NOTE: {note}")
    print(f"   intent: {response['intent']['emotion']} → {response['intent']['desired_tone']}; "
          f"keywords {response['intent']['keywords']}")
    for r in response["recommendations"]:
        print(f"   #{r['rank']} [{r['story_id']}] {r['final_score']:>5} {r['category']:<26} {r['title'][:55]}"
              f"{'  (placeholder)' if r['is_fallback_story'] else ''}")
    print("\n   Trace:")
    print_trace(response["trace"])
    print("\n   Metrics:")
    print_metrics(response["metrics"])

    if not response["recommendations"]:
        print("\n   No recommendations to open.")
        return response, checks
    sid = response["recommendations"][0]["story_id"]
    first = orch.get_story_details(sid)
    second = orch.get_story_details(sid)
    checks.append(("get_story_details() response is JSON-serializable", bool(json.dumps(first))))
    checks.append(("first details call is not a cache hit", first["cache_hit"] is False))
    checks.append(("second details call is a cache hit with identical story", second["cache_hit"] is True
                   and second["story"] == first["story"]))
    checks.append(("unknown story_id returns an error, no exception",
                   "error" in orch.get_story_details("does-not-exist")))
    story = first["story"]
    if story:
        checks.append(("summary is 70–90 words or explicitly flagged",
                       story["summary_passed"] or story["summary_status"] in ("extended", "trimmed", "out_of_range")))
        print(f"\n   get_story_details({sid!r}): {story['title'][:70]}")
        print(f"   summary ({story['word_count']} words, {story['summary_status']}): {story['summary'][:300]}")
        for note in story["validation_notes"]:
            print(f"   - {note}")
    print("\n   Trace (first call):")
    print_trace(first["trace"])
    print("   Trace (second call):")
    print_trace(second["trace"])
    print("\n   Metrics (first call):")
    print_metrics(first["metrics"])
    print("\n   Checks:")
    for name, ok in checks:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    return response, checks


orchestrator = StoryOrchestrator(llm)
print(f"orchestrator ready (backend: {'mock' if orchestrator.is_mock else 'real models'})")

# %% [markdown]
# ### End-to-end test, mock mode
#
# Always runs, even on a GPU runtime, using its own `MockModelManager`, so
# it finishes in seconds. It checks that:
# - `recommend()` returns at least 5 recommendations
# - both responses survive `json.dumps`
# - the first `get_story_details()` runs the Writer and Validator, and the
#   second call is a cache hit with an identical story
# - an unknown `story_id` returns an error instead of raising
# - the summary is 70–90 words or explicitly flagged
#
# Feeds and articles are real if the network is up. Otherwise the bundled
# fallback stories are used, and the notes say so.

# %%
mock_response, mock_checks = run_orchestrator_demo(
    StoryOrchestrator(MockModelManager(MODEL_IDS)), "stressed about exams, want something calming", "MOCK"
)

# %% [markdown]
# ### End-to-end test, real models (GPU only)
#
# Runs the same flow on the real models through the shared `orchestrator`,
# which the UI uses as well, so stories opened here are already cached for
# the UI. It is skipped on CPU runtimes or when `USE_MOCK_LLM` is on.
#
# On the first GPU run this cell includes model download and loading time,
# which shows up in the Mood agent's wall time.

# %%
if HAS_CUDA and not orchestrator.is_mock:
    real_response, real_checks = run_orchestrator_demo(orchestrator, "feeling lonely since moving to a new city", "REAL")
else:
    print("Skipped: real-model end-to-end test needs a CUDA GPU (USE_MOCK_LLM is on).")

# %% [markdown]
# ## 16. Evaluation (optional: `RUN_EVAL`)
#
# Runs **10 diverse sample moods** through the full pipeline without the UI,
# using a fresh `StoryOrchestrator`: the same code path the UI uses.
#
# **The moods.** They cover different emotions (stress, loneliness,
# boredom, grief, anxiety, joy, homesickness, curiosity) and lengths, down
# to a one-word "meh" that tests robustness.
#
# **What runs for each mood:**
# - `recommend()`: Mood → Retrieval → Classifier/Ranker, top 5
# - `get_story_details()` on the top `EVAL_STORIES_PER_MOOD` (2) stories:
#   Writer → Validator
#
# **Table 1, per mood:**
# - the detected emotion and the categories returned
# - whether the fallback stories were used
# - summaries within 70–90 words after validation
# - average summary retries
# - fact-check warnings, out of the stories written
# - URL success: shortlisted article URLs that resolved, out of those tried
# - time for the search and per opened story
# - total tokens in/out
#
# **Table 2, per stage (agent):** average latency per run, average tokens
# in/out per run, calls, and retries.
#
# **Overall summary:**
# - % of summaries within 70–90 words after validation. This is also split
#   into "first try", "after retries" and "fixed by trim/extend", so it's
#   clear how much the model did and how much the Validator's code did.
# - % of paraphrases flagged by the copy check, and % of stories with a
#   fact-check warning
# - category diversity and category-match rate
# - URL success rate and fallback usage
# - average tokens per mood
#
# **How to read it:**
# - **Process metrics only.** These numbers measure the pipeline, not
#   quality. There are no human ratings or ground-truth labels; see Known
#   Limitations.
# - **Caching.** Feeds and articles fetched earlier in this session come from
#   the 15-minute cache, so retrieval latency here is lower than a cold start.
#   If two moods share a story, its details come from the cache. Those cache
#   hits are counted, but excluded from latency and token averages and
#   counted once in the summary stats.
# - **Model loading.** On a GPU the models were already loaded by the smoke
#   test, so load time is not counted.
# - **Mock mode** measures plumbing only; the summary word counts are canned.
# - **Raw results** are kept in `eval_results` (JSON-safe) for the analysis
#   below.

# %%
EVAL_MOODS = [
    "stressed about exams next week",
    "feeling lonely since moving to a new city",
    "bored, want something fun about animals",
    "anxious about climate change after reading the news",
    "burnt out at work and need a reminder that people are kind",
    "just got a promotion and want to keep the good vibes going",
    "homesick international student missing my family",
    "my dog died last month and I'm still sad",
    "curious, want something amazing about science or space",
    "meh",
]


def _short_category(name):
    return name.split(" & ")[0]


def run_evaluation(orch, moods, stories_per_mood):
    per_mood, stage, seen_stories = [], {}, {}
    cache_hits = 0
    for mood in moods:
        t0 = time.perf_counter()
        try:
            rec = orch.recommend(mood, n=CLASSIFIER_TOP_N)
            details = [orch.get_story_details(r["story_id"]) for r in rec["recommendations"][:stories_per_mood]]
        except Exception as e:
            # One failing mood must not stop the evaluation; it is reported as a row with the error.
            per_mood.append({"mood": mood, "error": f"{type(e).__name__}: {str(e)[:150]}"})
            continue

        detail_seconds, tokens_in, tokens_out = [], 0, 0
        for resp in [rec] + details:
            is_cached = any(step["status"] == "cache_hit" for step in resp["trace"])
            cache_hits += is_cached
            if not is_cached:
                for step in resp["trace"]:
                    s = stage.setdefault(step["agent"], {"durations": [], "in": [], "out": [], "calls": 0, "retries": 0})
                    s["durations"].append(step["duration_s"])
                for agent, m in resp["metrics"]["per_agent"].items():
                    s = stage.setdefault(agent, {"durations": [], "in": [], "out": [], "calls": 0, "retries": 0})
                    s["in"].append(m["input_tokens"])
                    s["out"].append(m["output_tokens"])
                    s["calls"] += m["calls"]
                    s["retries"] += m["retries"]
            tokens_in += resp["metrics"]["total"]["input_tokens"]
            tokens_out += resp["metrics"]["total"]["output_tokens"]
            if resp is not rec and not is_cached:
                detail_seconds.append(resp["metrics"]["total"]["wall_time_s"])

        written = [d["story"] for d in details if d.get("story")]
        for story in written:
            seen_stories.setdefault(story["story_id"], story)
        stats = rec["retrieval_stats"]
        per_mood.append({
            "mood": mood,
            "emotion": rec["intent"]["emotion"],
            "intent_parse": rec["intent"]["parse_method"],
            "categories": sorted({r["category"] for r in rec["recommendations"]}),
            "n_recommendations": len(rec["recommendations"]),
            "category_matches": sum(r["category_match"] for r in rec["recommendations"]),
            "rejected": len(rec["rejected"]),
            "used_fallback": rec["used_fallback"],
            "summaries_written": len(written),
            "summaries_in_range": sum(s["summary_passed"] for s in written),
            "facts_flagged": sum(s["facts_flagged"] for s in written),
            "avg_retries": (sum(len(s["summary_attempt_word_counts"]) - 1 for s in written) / len(written)
                            if written else None),
            "urls_attempted": stats["urls_attempted"],
            "urls_resolved": stats["urls_resolved"],
            "search_seconds": rec["metrics"]["total"]["wall_time_s"],
            "avg_detail_seconds": sum(detail_seconds) / len(detail_seconds) if detail_seconds else None,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "wall_seconds": round(time.perf_counter() - t0, 2),
        })

    stories = list(seen_stories.values())
    ok = [r for r in per_mood if "error" not in r]

    def pct(part, whole):
        return round(100 * part / whole, 1) if whole else None

    statuses = [s["summary_status"] for s in stories]
    n_recs = sum(r["n_recommendations"] for r in ok)
    attempted = sum(r["urls_attempted"] for r in ok)
    summary = {
        "moods_evaluated": len(ok),
        "moods_failed": len(per_mood) - len(ok),
        "stories_written": len(stories),
        "details_cache_hits": cache_hits,
        "pct_summaries_in_range_after_validation": pct(sum(s["summary_passed"] for s in stories), len(stories)),
        "pct_in_range_first_try": pct(statuses.count("ok"), len(stories)),
        "pct_in_range_after_retries": pct(statuses.count("ok_after_retry"), len(stories)),
        "pct_fixed_by_trim_or_extend": pct(statuses.count("trimmed") + statuses.count("extended"), len(stories)),
        "pct_out_of_range": pct(statuses.count("out_of_range"), len(stories)),
        "avg_summary_retries": (round(sum(len(s["summary_attempt_word_counts"]) - 1 for s in stories) / len(stories), 2)
                                if stories else None),
        "pct_copy_flagged": pct(sum(s["copy_flagged"] for s in stories), len(stories)),
        "pct_facts_flagged": pct(sum(s["facts_flagged"] for s in stories), len(stories)),
        "url_success_rate_pct": pct(sum(r["urls_resolved"] for r in ok), attempted),
        "moods_using_fallback": sum(r["used_fallback"] for r in ok),
        "distinct_categories": len({c for r in ok for c in r["categories"]}),
        "pct_recs_matching_target_category": pct(sum(r["category_matches"] for r in ok), n_recs),
        "intent_parse_methods": {m: sum(r["intent_parse"] == m for r in ok) for m in ("json", "regex", "default")},
        "avg_tokens_in_per_mood": round(sum(r["tokens_in"] for r in ok) / len(ok)) if ok else None,
        "avg_tokens_out_per_mood": round(sum(r["tokens_out"] for r in ok) / len(ok)) if ok else None,
    }
    per_stage = {
        agent: {
            "runs": len(s["durations"]),
            "avg_latency_s": round(sum(s["durations"]) / len(s["durations"]), 3) if s["durations"] else None,
            "avg_tokens_in": round(sum(s["in"]) / len(s["in"])) if s["in"] else 0,
            "avg_tokens_out": round(sum(s["out"]) / len(s["out"])) if s["out"] else 0,
            "calls": s["calls"],
            "retries": s["retries"],
        }
        for agent, s in sorted(stage.items(), key=lambda kv: (SEARCH_AGENTS + DETAIL_AGENTS).index(kv[0])
                               if kv[0] in SEARCH_AGENTS + DETAIL_AGENTS else 99)
    }
    return to_jsonable({"mock_llm": orch.is_mock, "stories_per_mood": stories_per_mood,
                        "per_mood": per_mood, "per_stage": per_stage, "summary": summary})


def print_evaluation(results):
    print(f"{'#':>2} {'mood':<34} {'emotion':<11} {'categories returned':<38} {'fb':>2} {'in 70-90':>8} "
          f"{'retries':>7} {'fact warn':>9} {'URLs ok':>8} {'search s':>8} {'story s':>7} {'tok in/out':>12}")
    for i, r in enumerate(results["per_mood"], 1):
        if "error" in r:
            print(f"{i:>2} {_clip(r['mood'], 34):<34} ERROR: {r['error']}")
            continue
        cats = ", ".join(_short_category(c) for c in r["categories"])
        retries = f"{r['avg_retries']:.1f}" if r["avg_retries"] is not None else "–"
        story_s = f"{r['avg_detail_seconds']:.1f}" if r["avg_detail_seconds"] is not None else "–"
        print(f"{i:>2} {_clip(r['mood'], 34):<34} {_clip(r['emotion'], 11):<11} {_clip(cats, 38):<38} "
              f"{'Y' if r['used_fallback'] else '':>2} {str(r['summaries_in_range']) + '/' + str(r['summaries_written']):>8} "
              f"{retries:>7} {str(r['facts_flagged']) + '/' + str(r['summaries_written']):>9} "
              f"{str(r['urls_resolved']) + '/' + str(r['urls_attempted']):>8} "
              f"{r['search_seconds']:>8.1f} {story_s:>7} {str(r['tokens_in']) + '/' + str(r['tokens_out']):>12}")

    print(f"\n{'stage':<18} {'runs':>5} {'avg latency s':>13} {'avg tok in':>10} {'avg tok out':>11} {'calls':>6} {'retries':>7}")
    for agent, s in results["per_stage"].items():
        latency = f"{s['avg_latency_s']:.2f}" if s["avg_latency_s"] is not None else "–"
        print(f"{AGENT_LABELS.get(agent, agent):<18} {s['runs']:>5} {latency:>13} {s['avg_tokens_in']:>10} "
              f"{s['avg_tokens_out']:>11} {s['calls']:>6} {s['retries']:>7}")

    s = results["summary"]
    print("\nSummary")
    print(f"  moods evaluated: {s['moods_evaluated']} (failed: {s['moods_failed']}); stories written: "
          f"{s['stories_written']} (+{s['details_cache_hits']} details cache hits)")
    print(f"  summaries within 70-90 words after validation: {s['pct_summaries_in_range_after_validation']}%  "
          f"[first try {s['pct_in_range_first_try']}% · after retries {s['pct_in_range_after_retries']}% · "
          f"fixed by trim/extend {s['pct_fixed_by_trim_or_extend']}% · out of range {s['pct_out_of_range']}%]")
    print(f"  average summary retries per story: {s['avg_summary_retries']}; paraphrases copy-flagged: "
          f"{s['pct_copy_flagged']}%; stories with a fact-check warning: {s['pct_facts_flagged']}%")
    print(f"  URL success rate: {s['url_success_rate_pct']}%; moods that needed the fallback stories: "
          f"{s['moods_using_fallback']}/{s['moods_evaluated']}")
    print(f"  distinct categories returned: {s['distinct_categories']} of {len(CATEGORIES)}; recommendations in a "
          f"target category: {s['pct_recs_matching_target_category']}%; intent parse methods: {s['intent_parse_methods']}")
    print(f"  average tokens per mood (search + {results['stories_per_mood']} stories): "
          f"{s['avg_tokens_in_per_mood']} in / {s['avg_tokens_out_per_mood']} out")


if RUN_EVAL:
    print(f"Evaluating {len(EVAL_MOODS)} moods × (search + {EVAL_STORIES_PER_MOOD} stories) on "
          f"{'MOCK outputs (plumbing only)' if orchestrator.is_mock else 'the real models'}. "
          "Set RUN_EVAL = False in the config cell to skip.\n")
    _eval_start = time.perf_counter()
    eval_results = run_evaluation(StoryOrchestrator(llm), EVAL_MOODS, EVAL_STORIES_PER_MOOD)
    print_evaluation(eval_results)
    print(f"\nEvaluation wall time: {time.perf_counter() - _eval_start:.1f}s")
else:
    eval_results = None
    print("Evaluation skipped (RUN_EVAL = False).")

# %% [markdown]
# ### Results & Analysis
#
# *Fill this in from the evaluation output above after a run on a T4. Each
# `[FILL]` is a number or a short observation. Delete this note when done.*
#
# **Setup.** Run on `[FILL: GPU, e.g. Tesla T4]` with `[FILL: real models / mock]`, `[FILL: date]`.
# 10 moods, top `[FILL: 2]` stories written per mood. Total evaluation time: `[FILL] s`.
#
# **Summary-length compliance.**
# - `[FILL]%` of summaries were within 70–90 words after validation.
# - Of those, `[FILL]%` passed on the first try, `[FILL]%` after Validator
#   retries, and `[FILL]%` needed a programmatic trim or extend.
# - Average retries per story: `[FILL]`.
# - *Observation:* `[FILL: e.g. does the 7B tend to run long or short?]`
#
# **Retrieval.**
# - URL success rate: `[FILL]%`.
# - The fallback stories were needed for `[FILL]`/10 moods.
# - Feeds that failed: `[FILL: from the feed health check]`.
# - *Observation:* `[FILL: e.g. which moods got weak keyword matches?]`
#
# **Classification and relevance.**
# - `[FILL]` distinct categories returned.
# - `[FILL]%` of recommendations were in a category the reader asked for.
# - `[FILL]` stories were rejected as not uplifting.
# - Spot check: `[FILL: 2–3 examples of good and bad picks, e.g. for "my dog died last month"]`.
#
# **Latency and cost.**
# - Average latency per stage: Mood `[FILL]` s, Retrieval `[FILL]` s,
#   Classifier `[FILL]` s, Writer `[FILL]` s per story, Validator `[FILL]` s.
# - Average tokens per mood: `[FILL]` in / `[FILL]` out.
# - Peak GPU memory: `[FILL]` GB, from the Metrics panel or the smoke test.
# - *Bottleneck:* `[FILL: e.g. the Writer's paraphrase call]`.
#
# **Robustness.**
# - Intent parse methods: `[FILL: json / regex / default counts]`.
# - The one-word mood "meh" produced `[FILL]`.
# - Copy-check flags: `[FILL]%`. Fact-check warnings: `[FILL]%` of stories.
#
# **Takeaways.** `[FILL: 2–3 sentences: what worked, what didn't, and what you would change first]`

# %% [markdown]
# ### Known Limitations
#
# **Retrieval quality**
# - **Simple ranking.** Candidates are ranked by keyword overlap with RSS
#   titles and summaries, not by meaning, so a mood whose words don't appear
#   in headlines gets weaker picks. A substring hit can also mislead: "rescue"
#   matched a story about repairing stuffed animals.
# - **Few sources.** Only 5 English-language positive-news feeds are used,
#   and each exposes only its latest 10–50 items. Recall is small, and
#   coverage depends on the day.
# - **Sites can block us.** Some sites refuse the notebook's honest bot
#   User-Agent: Good News Network sits behind Cloudflare, which returned 403
#   for some pages, and at one point for the whole feed, during development.
#   Datacenter IPs like Colab's may be blocked more often. When fewer than 5
#   live stories survive, built-in placeholder stories fill in. They are
#   clearly labeled, but they are not real news.
# - **Imperfect extraction.** trafilatura sometimes keeps boilerplate
#   ("share this story…", donation appeals) or picks up podcast transcripts.
#   Articles are cut to 6,000 characters, and the Writer only sees the first
#   4,000.
# - **One category per story.** Each story gets exactly one category, but
#   many fit several (a neighbor's act of kindness is also Community Impact),
#   so the label can look arbitrary between close candidates.
#
# **Model weaknesses**
# - **Uplifting judgments.** The Classifier (Phi-3.5-mini, 4-bit) can call a
#   tragedy-with-a-silver-lining story "uplifting". A smaller stand-in model
#   did exactly that in testing. Quantization may cost some accuracy too.
# - **Shallow fact-checking.** The Validator checks that numbers and
#   multi-word names in the paraphrase and summary appear in the source, and
#   warns if they don't. Invented places, single-word names, changed
#   descriptions and real facts used wrongly still get through (see the fact
#   check limits in section 13).
# - **Lexical copy check.** It catches copied word runs, but not close
#   paraphrase that copies sentence structure. It can also flag legitimately
#   repeated names and titles.
# - **Word-count fixes are mechanical.** The 70–90 word rule is guaranteed
#   by code, not by the model. When retries fail, the summary is trimmed at
#   a sentence boundary or extended with sentences from the paraphrase or
#   source. It always lands in range, but can read less smoothly, and it is
#   flagged in the UI.
# - **Intent depends on the small model.** Vague moods ("meh") fall back to
#   generic keywords.
# - **Sampling varies.** The Writer samples (temperature 0.7), so repeated
#   runs give different text and different retry counts.
#
# **Latency on a free T4**
# - **Slow hardware.** The T4 has no bf16 or FlashAttention-2 support, and
#   4-bit bitsandbytes inference de-quantizes on the fly, so generation is
#   much slower than on newer GPUs. The Writer's paraphrase plus summary
#   (and any retries) per story is the slowest step. See Results & Analysis
#   for measured numbers.
# - **Slow first run.** The first run downloads about 25 GB of weights. If
#   the GPU runs out of memory, the manager switches to loading one model at
#   a time, which then reloads models on every request.
# - **One request at a time,** with no streaming of the generated text
#   itself. Only the agent trace updates live.
#
# **Hosting and the Gradio link**
# - **Temporary link.** The `share=True` link expires: Gradio prints the
#   expiry when it launches (**UNVERIFIED:** about a week in recent
#   versions). It also dies as soon as the Colab runtime stops, idles out
#   or is recycled, since free Colab sessions have usage limits.
# - **No access control.** Anyone with the link can use the app, and it has
#   no authentication or rate limiting.
# - **In-memory state.** Caches (feeds, articles, written stories) are lost
#   on restart. A cached story keeps the tone of the mood that first
#   produced it.
#
# **Evaluation**
# - **Process metrics only.** The metrics above measure compliance, latency,
#   tokens and robustness, not relevance or writing quality. There are no
#   human ratings or reference summaries.
# - **Small sample.** 10 moods and 2 stories each is enough to spot problems,
#   not to estimate rates precisely.

# %% [markdown]
# ## 17. Gradio UI: rendering helpers
#
# Small functions that turn the orchestrator's JSON into HTML for the UI.
#
# - **Story cards:** rank, title, source, category badge, a mood-fit bar,
#   and the classifier's one-line reason.
# - **Story detail panel:**
#   - title, source and a clickable link, with the category badge
#   - the paraphrase
#   - the summary, with a word-count pill: green if it passed 70–90, amber if
#     the Validator had to trim or extend it, red if out of range
#   - copy-check and fact-check tags next to the category badge. The fact
#     check tag lists up to 3 names or numbers that were not found in the
#     source, and turns amber when the Validator raised a warning
#   - the validation notes
# - **Agent Trace timeline:** one row per agent step, with a colored status
#   dot and label, model, duration and output preview. Each row expands to show the
#   agent's detailed events. Steps that haven't run yet appear as pending,
#   and the current one as running, which is how the trace updates live.
# - **Metrics:** headline tiles (input/output tokens, time, tokens/sec, GPU
#   memory) and a per-agent table. Search metrics and story-detail metrics
#   are combined into one full-pipeline view.
#
# **Safety:** article titles, reasons and paraphrases come from the web and
# from model output, and `gr.HTML` does not sanitize. So every dynamic string
# is HTML-escaped, and links are only rendered for `http(s)` URLs.
#
# **Theme:** the app uses the community theme `allenai/gradio-theme` from the
# Hugging Face Hub (light and dark mode), with Gradio's built-in Soft theme as
# a fallback if the download fails. The cards take their colors from the
# theme's CSS variables, so they follow it in both modes.

# %%
from urllib.parse import urlsplit as _urlsplit

esc = html_lib.escape


def safe_url(url):
    return url if isinstance(url, str) and _urlsplit(url).scheme in ("http", "https") else None


def render_banner(kind, text):
    # kind: "info" | "warn" | "error" | "mock" | "ok"
    return f'<div class="sd-banner sd-{kind}">{text}</div>' if text else ""


def render_notes(response):
    parts = []
    if response.get("mock_llm"):
        parts.append(render_banner("mock", "<b>Mock mode</b>: model outputs are canned placeholders so the "
                                           "app can be tested without a GPU. Feeds and articles are still real."))
    if response.get("used_fallback"):
        parts.append(render_banner("warn", "<b>Offline fallback used</b>: live feeds returned too few usable "
                                           "stories, so some results are built-in placeholder stories, not real "
                                           "news. They are tagged <i>placeholder</i> below."))
    for note in response.get("notes", []):
        if "MOCK MODE" in note or "placeholder" in note:
            continue  # already covered by the banners above
        parts.append(render_banner("info", esc(note)))
    if response.get("preferred_source"):
        parts.append(render_banner("info", f"Preferred source: <b>{esc(response['preferred_source'])}</b> "
                                           "(its stories are ranked first)."))
    return "".join(parts)


def render_cards(recommendations):
    if not recommendations:
        return render_banner("warn", "No uplifting stories matched this time. Try a different mood or source.")
    cards = []
    for r in recommendations:
        fit = max(0, min(10, int(r.get("mood_fit_score") or 0)))
        date = (r.get("published") or "")[:10]
        meta = " · ".join(p for p in [esc(r["source"]), esc(date)] if p)
        tags = '<span class="sd-tag">placeholder</span>' if r.get("is_fallback_story") else ""
        tags += '<span class="sd-tag sd-tag-match">matches your mood</span>' if r.get("category_match") else ""
        cards.append(f"""
<div class="sd-card">
  <div class="sd-card-top"><span class="sd-rank">{r['rank']}</span><span class="sd-badge">{esc(r['category'])}</span>{tags}</div>
  <div class="sd-card-title">{esc(r['title'])}</div>
  <div class="sd-muted">{meta}</div>
  <div class="sd-fit" title="Mood fit {fit}/10"><div class="sd-fit-bar"><span style="width:{fit * 10}%"></span></div>
    <span class="sd-fit-label">{fit}/10 mood fit</span></div>
  <div class="sd-card-reason">{esc(r.get('reason') or '')}</div>
</div>""")
    return f'<div class="sd-cards">{"".join(cards)}</div>'


def story_choices(recommendations):
    # Short labels: the cards below carry the full details, and the numbers match the cards' rank badges.
    return [(f"{r['rank']}. {_clip(r['title'], 42)}", r["story_id"]) for r in recommendations]


DETAIL_PLACEHOLDER = ('<div class="sd-detail sd-empty">Pick a story from the list to read a fresh paraphrase '
                      'and a 70–90 word summary.</div>')


def render_detail_loading(title):
    return (f'<div class="sd-detail sd-empty">Writing a paraphrase and summary for <b>{esc(title)}</b>…'
            '<br><span class="sd-muted">The Writer runs only for stories you open; opening it again is instant.'
            '</span></div>')


def render_detail(story):
    url = safe_url(story.get("url"))
    link = (f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">Read the original ↗</a>'
            if url else '<span class="sd-muted">(no link: built-in placeholder story)</span>')
    date = (story.get("published") or "")[:10]
    status = story.get("summary_status", "")
    if status in ("ok", "ok_after_retry"):
        pill_class, pill_note = "sd-pill-ok", "within 70–90"
    elif status in ("trimmed", "extended"):
        pill_class, pill_note = "sd-pill-warn", f"{status} by the Validator"
    else:
        pill_class, pill_note = "sd-pill-bad", "outside 70–90"
    attempts = story.get("summary_attempt_word_counts") or []
    attempts_note = f" · attempts {attempts}" if len(attempts) > 1 else ""
    paragraphs = "".join(f"<p>{esc(p)}</p>" for p in (story.get("paraphrase") or "").split("\n\n") if p.strip())
    paraphrase = paragraphs or '<p class="sd-muted">The Writer produced no paraphrase for this story.</p>'
    copy_tag = ('<span class="sd-tag sd-tag-bad">copy check flagged</span>' if story.get("copy_flagged")
                else '<span class="sd-tag">copy check passed</span>')
    facts = [story.get("paraphrase_facts") or {}, story.get("summary_facts") or {}]
    missing = list(dict.fromkeys(m for f in facts for m in f.get("unsupported_numbers", []) + f.get("unsupported_names", [])))
    if missing:
        shown = ", ".join(f"'{m}'" for m in missing[:3]) + (", …" if len(missing) > 3 else "")
        facts_tag = (f'<span class="sd-tag {"sd-tag-warn" if story.get("facts_flagged") else ""}" '
                     f'title="Names or numbers not found in the source article">facts check: {len(missing)} '
                     f'unverified: {esc(shown)}</span>')
    elif any(f.get("checked") for f in facts):
        facts_tag = '<span class="sd-tag">facts check passed</span>'
    else:
        facts_tag = '<span class="sd-tag">facts check: no names or numbers</span>'
    fallback = render_banner("warn", "This is a built-in placeholder story (offline fallback), not real news."
                             ) if story.get("is_fallback_story") else ""
    notes = "".join(f"<li>{esc(n)}</li>" for n in story.get("validation_notes", []))
    return f"""
<div class="sd-detail">
  {fallback}
  <div class="sd-card-top"><span class="sd-badge">{esc(story.get('category', ''))}</span>{copy_tag}{facts_tag}</div>
  <h2 class="sd-detail-title">{esc(story.get('title', ''))}</h2>
  <div class="sd-muted">{esc(story.get('source', ''))}{' · ' + esc(date) if date else ''} · {link}</div>
  <h3>Summary <span class="sd-pill {pill_class}">{story.get('word_count', 0)} words · {esc(pill_note)}</span></h3>
  <p class="sd-summary">{esc(story.get('summary', ''))}</p>
  <div class="sd-muted sd-small">Counted by the Validator{esc(attempts_note)}</div>
  <h3>Paraphrased version <span class="sd-muted sd-small">({story.get('paraphrase_word_count', 0)} words, in the model's own words)</span></h3>
  <div class="sd-paraphrase">{paraphrase}</div>
  <details class="sd-notes"><summary>Validation notes ({len(story.get('validation_notes', []))})</summary><ul>{notes}</ul></details>
</div>"""


def render_trace(steps, pending_agents=(), running=None):
    rows = []
    for s in steps:
        events = "".join(f"<li><code>{e['elapsed_s']:.2f}s</code> {'<b>Warning:</b> ' if e['fallback_used'] else ''}"
                         f"{esc(e['message'])}</li>" for e in s.get("events", []))
        rows.append(f"""
<details class="sd-step sd-status-{esc(s['status'])}">
  <summary><span class="sd-step-icon"></span>
    <b>{esc(AGENT_LABELS.get(s['agent'], s['agent']))}</b>
    <span class="sd-muted">{esc(s['model'])}</span>
    <span class="sd-step-status">{esc(s['status'].replace('_', ' '))}</span>
    <span class="sd-step-time">{s['duration_s']:.2f}s</span>
    <div class="sd-step-preview">{esc(s['output_preview'])}</div></summary>
  <ul class="sd-events">{events or '<li class="sd-muted">No detailed events.</li>'}</ul>
</details>""")
    for agent in pending_agents:
        state = "running" if agent == running else "pending"
        label = "running…" if state == "running" else "waiting"
        rows.append(f"""
<div class="sd-step sd-status-{state}"><span class="sd-step-icon"></span>
  <b>{esc(AGENT_LABELS[agent])}</b> <span class="sd-step-status">{label}</span></div>""")
    if not rows:
        return '<div class="sd-muted">The agent steps will appear here as each agent runs.</div>'
    return f'<div class="sd-trace">{"".join(rows)}</div>'


def combine_metrics(*metric_sets, mock=False):
    per_agent = {}
    for metrics in metric_sets:
        if metrics:
            per_agent.update(metrics["per_agent"])
    return {"per_agent": per_agent, "total": aggregate_totals(per_agent, rates_meaningful=not mock)}


def render_metric_tiles(metrics, mock=False):
    t = metrics["total"]
    gpu = "n/a (CPU)"
    if t.get("peak_gpu_memory_gb") is not None or t.get("gpu_memory_allocated_gb") is not None:
        peak = f"{t['peak_gpu_memory_gb']:.2f}" if t.get("peak_gpu_memory_gb") is not None else "–"
        gpu = f"{peak} GB peak · {t.get('gpu_memory_allocated_gb') or 0:.2f} GB now"
    tiles = [
        ("Input tokens", f"{t['input_tokens']:,}"), ("Output tokens", f"{t['output_tokens']:,}"),
        ("Total time", f"{t['wall_time_s']:.1f} s"),
        ("Tokens / sec", "n/a (mock)" if mock else f"{t['tokens_per_second']:.1f}"),
        ("GPU memory", gpu), ("Retries · cache hits", f"{t['retries']} · {t['cache_hits']}"),
    ]
    return '<div class="sd-tiles">' + "".join(
        f'<div class="sd-tile"><div class="sd-tile-label">{esc(k)}</div><div class="sd-tile-value">{esc(v)}</div></div>'
        for k, v in tiles) + "</div>"


METRIC_HEADERS = ["Agent", "Calls", "Input tok", "Output tok", "Wall s", "Tok/s", "Peak GPU GB", "Retries", "Cache hits"]


def metrics_rows(metrics):
    order = SEARCH_AGENTS + DETAIL_AGENTS
    rows = []
    for agent in sorted(metrics["per_agent"], key=lambda a: order.index(a) if a in order else 99):
        m = metrics["per_agent"][agent]
        rows.append([AGENT_LABELS.get(agent, agent), m["calls"], m["input_tokens"], m["output_tokens"],
                     round(m["wall_time_s"], 2), m["tokens_per_second"],
                     m["peak_gpu_memory_gb"] if m["peak_gpu_memory_gb"] is not None else "n/a",
                     m["retries"], m["cache_hits"]])
    t = metrics["total"]
    if rows:
        rows.append(["TOTAL", t["calls"], t["input_tokens"], t["output_tokens"], round(t["wall_time_s"], 2),
                     t["tokens_per_second"], t["peak_gpu_memory_gb"] if t["peak_gpu_memory_gb"] is not None else "n/a",
                     t["retries"], t["cache_hits"]])
    return rows


def friendly_error(action, error):
    return render_banner("error", f"Something went wrong while {esc(action)}. You can try again or pick "
                                  f"another mood.<br><span class='sd-small'>Details: {esc(type(error).__name__)}: "
                                  f"{esc(str(error)[:200])}</span>")

# %% [markdown]
# ## 18. Gradio UI: the app
#
# Layout, top to bottom:
#
# 1. **Header** with the project name, a tagline, and a pill showing whether
#    real models (with GPU name) or mock mode are active.
# 2. **Inputs:**
#    - a mood textbox with example-mood chips (click to fill)
#    - a source dropdown (default "Any", otherwise one of the configured
#      feeds)
#    - the **Tell Me a Story** button. Pressing Enter in the textbox works
#      too.
# 3. **Results** on the left: banners (mock mode, offline fallback, notes),
#    then a compact story picker numbered like the cards, then the 5 story
#    cards. **Story detail** on the right: it loads when a story is chosen.
# 4. **Agent Trace** accordion, open by default, updated live as each agent
#    finishes.
# 5. **Metrics** accordion: headline tiles plus the per-agent breakdown.
#
# **How "live" works.** The click handlers are Python generators. Each one
# runs the orchestrator in a worker thread and passes an `on_step` callback
# that puts every finished step on a queue. The handler yields a fresh trace,
# with the remaining agents shown as running or pending, each time a step
# arrives. Then it yields the final results.
#
# **Loading and errors:**
# - While a request runs, the button is disabled and relabelled, and the
#   panels show a loading message.
# - An empty mood gets a friendly prompt instead of a search.
# - Any exception becomes a friendly error banner, and the app keeps
#   working.
#
# **One request at a time** (`default_concurrency_limit=1` in the launch
# cell). There is a single GPU, and the orchestrator swaps the agents' model
# backend while it runs.

# %%
import queue as queue_lib

import gradio as gr

UI_EXAMPLE_MOODS = [
    "stressed about exams",
    "feeling lonely since moving to a new city",
    "bored, want something fun about animals",
    "worn out by bad news, need some hope",
    "proud of my kid and want more good vibes",
]
SOURCE_CHOICES = ["Any"] + list(RSS_SOURCES)

# UNVERIFIED: on a Colab runtime, that `allenai/gradio-theme` downloads from the Hub (it loaded on the dev Mac).
try:
    UI_THEME = gr.Theme.from_hub("allenai/gradio-theme")
except Exception as e:
    # A theme download failure must not stop "Run all" before the UI launches.
    print(f"Could not load the Hub theme ({type(e).__name__}); using Gradio's built-in Soft theme.")
    UI_THEME = gr.themes.Soft(primary_hue="pink", neutral_hue="slate", radius_size="lg")

UI_CSS = """
/* width:100% matters: with auto margins alone the container sizes to its content, and the wide metrics
   table then stretches the whole page past a phone's screen. */
.gradio-container { max-width: 1240px !important; width: 100% !important; margin: 0 auto !important; }
.sd-header { padding: 18px 4px 6px; }
.sd-header h1 { font-size: 2rem; margin: 0; letter-spacing: -0.02em; }
.sd-header p { margin: 4px 0 0; color: var(--body-text-color-subdued); font-size: 1.05rem; }
.sd-mode { display: inline-block; margin-top: 10px; padding: 3px 10px; border-radius: 999px; font-size: .8rem;
  border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); }
.sd-chips { flex-wrap: wrap !important; gap: 8px !important; }
button.sd-chip { border-radius: 999px !important; font-size: .85rem !important; white-space: nowrap !important;
  width: auto !important; min-width: 0 !important; flex: 0 0 auto !important; padding: 4px 14px !important; }
#sd-pick label { font-size: .88rem; }
/* Without wrapping, the two 320px-min columns force a ~740px page on phones (horizontal scroll). */
.sd-split { flex-wrap: wrap !important; }
#sd-go { font-size: 1.1rem !important; font-weight: 700 !important; min-height: 48px; }
.sd-banner { padding: 10px 14px; border-radius: 10px; margin: 6px 0; font-size: .92rem; border: 1px solid; }
.sd-info { border-color: var(--border-color-primary); background: var(--block-background-fill); }
.sd-warn { border-color: #d97706; background: color-mix(in srgb, #f59e0b 12%, transparent); }
.sd-error { border-color: #dc2626; background: color-mix(in srgb, #ef4444 12%, transparent); }
.sd-mock { border-color: #7c3aed; background: color-mix(in srgb, #8b5cf6 12%, transparent); }
.sd-ok { border-color: #16a34a; background: color-mix(in srgb, #22c55e 12%, transparent); }
.sd-cards { display: grid; gap: 10px; }
.sd-card { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px;
  background: var(--block-background-fill); }
.sd-card-top { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 6px; }
.sd-rank { font-weight: 700; width: 24px; height: 24px; border-radius: 50%; display: inline-flex;
  align-items: center; justify-content: center; background: var(--color-accent); color: white; font-size: .8rem; }
.sd-badge { padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  background: color-mix(in srgb, var(--color-accent) 16%, transparent); color: var(--body-text-color); }
.sd-tag { padding: 2px 8px; border-radius: 999px; font-size: .72rem; border: 1px solid var(--border-color-primary);
  color: var(--body-text-color-subdued); }
.sd-tag-match { border-color: #16a34a; color: #16a34a; }
.sd-tag-bad { border-color: #dc2626; color: #dc2626; }
.sd-tag-warn { border-color: #d97706; color: #d97706; }
.sd-card-title { font-weight: 650; font-size: 1.02rem; line-height: 1.3; }
.sd-muted { color: var(--body-text-color-subdued); font-size: .85rem; }
.sd-small { font-size: .78rem; }
.sd-fit { display: flex; align-items: center; gap: 8px; margin: 8px 0 4px; }
.sd-fit-bar { flex: 1; height: 8px; border-radius: 999px; background: var(--border-color-primary); overflow: hidden; }
.sd-fit-bar span { display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, color-mix(in srgb, var(--color-accent) 45%, transparent), var(--color-accent)); }
.sd-fit-label { font-size: .78rem; font-weight: 600; white-space: nowrap; }
.sd-card-reason { font-size: .85rem; color: var(--body-text-color-subdued); font-style: italic; }
.sd-detail { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 16px 18px;
  background: var(--block-background-fill); min-height: 200px; }
.sd-empty { display: flex; flex-direction: column; justify-content: center; text-align: center;
  color: var(--body-text-color-subdued); }
.sd-detail-title { margin: 6px 0 4px; font-size: 1.35rem; line-height: 1.3; }
.sd-detail h3 { margin: 18px 0 6px; font-size: 1rem; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.sd-summary { font-size: 1.02rem; line-height: 1.6; margin: 0; }
.sd-paraphrase p { line-height: 1.6; margin: 0 0 10px; }
.sd-pill { padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; border: 1px solid; }
.sd-pill-ok { border-color: #16a34a; color: #16a34a; }
.sd-pill-warn { border-color: #d97706; color: #d97706; }
.sd-pill-bad { border-color: #dc2626; color: #dc2626; }
.sd-notes { margin-top: 12px; font-size: .85rem; }
.sd-trace { display: grid; gap: 6px; }
.sd-step { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 8px 12px;
  background: var(--block-background-fill); }
.sd-step summary { cursor: pointer; list-style: none; }
/* A colored dot per status: the status word next to it carries the meaning, the color makes it scannable. */
.sd-step-icon { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px;
  vertical-align: middle; background: var(--border-color-primary); }
.sd-status-ok .sd-step-icon { background: #16a34a; }
.sd-status-fallback .sd-step-icon { background: #d97706; }
.sd-status-error .sd-step-icon { background: #dc2626; }
.sd-status-running .sd-step-icon, .sd-status-cache_hit .sd-step-icon { background: var(--color-accent); }
.sd-step-status { margin-left: 8px; font-size: .75rem; text-transform: uppercase; letter-spacing: .04em;
  color: var(--body-text-color-subdued); }
.sd-step-time { float: right; font-variant-numeric: tabular-nums; font-size: .85rem; }
.sd-step-preview { font-size: .82rem; color: var(--body-text-color-subdued); margin-top: 2px; }
.sd-status-running { border-color: var(--color-accent); }
.sd-status-pending { opacity: .55; }
.sd-status-fallback { border-color: #d97706; }
.sd-status-error { border-color: #dc2626; }
.sd-events { font-size: .8rem; margin: 8px 0 0; padding-left: 18px; }
.sd-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-bottom: 8px; }
.sd-tile { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 12px;
  background: var(--block-background-fill); }
.sd-tile-label { font-size: .75rem; color: var(--body-text-color-subdued); text-transform: uppercase; letter-spacing: .04em; }
.sd-tile-value { font-size: 1.15rem; font-weight: 700; font-variant-numeric: tabular-nums; margin-top: 2px; }
"""


def _mode_label():
    if orchestrator.is_mock:
        return "Mock mode: canned model outputs (no GPU needed)"
    gpu = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU"
    return f"Real models on {gpu}"


def _stream(call):
    """Run call(on_step) in a worker thread; yield ("step", steps_so_far) per finished step, then ("done", result)."""
    q = queue_lib.Queue()
    outcome = {}

    def worker():
        try:
            outcome["result"] = call(q.put)
        except Exception as e:
            outcome["error"] = e
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    steps = []
    while True:
        item = q.get()
        if item is None:
            break
        steps.append(item)
        yield "step", steps
    if "error" in outcome:
        raise outcome["error"]
    yield "done", outcome["result"]


def _pending(agents, steps):
    done = {s["agent"] for s in steps}
    remaining = [a for a in agents if a not in done]
    return remaining, (remaining[0] if remaining else None)


def on_search(mood, source):
    no_change = gr.update()
    mood = (mood or "").strip()
    if not mood:
        yield (render_banner("warn", "Tell me a little about how you're feeling first, or tap one of the "
                                     "example moods."), no_change, no_change, no_change, no_change, no_change,
               no_change, no_change, no_change)
        return
    preferred = None if source in (None, "", "Any") else source
    busy_btn = gr.update(value="Finding stories…", interactive=False)
    idle_btn = gr.update(value="Tell Me a Story", interactive=True)
    yield (render_banner("info", f"Finding uplifting stories for “{esc(mood)}”…"), "",
           gr.update(choices=[], value=None, visible=False), DETAIL_PLACEHOLDER,
           render_trace([], SEARCH_AGENTS, "mood"), no_change, no_change, busy_btn, None)
    try:
        for kind, payload in _stream(lambda cb: orchestrator.recommend(mood, n=CLASSIFIER_TOP_N,
                                                                       preferred_source=preferred, on_step=cb)):
            if kind == "step":
                remaining, running = _pending(SEARCH_AGENTS, payload)
                yield (no_change, no_change, no_change, no_change, render_trace(payload, remaining, running),
                       no_change, no_change, busy_btn, no_change)
                continue
            response = payload
            recs = response["recommendations"]
            metrics = combine_metrics(response["metrics"], mock=response["mock_llm"])
            status = render_notes(response)
            if recs:
                status += render_banner("ok", f"Found {len(recs)} stories for you. Pick one to read its summary.")
            yield (status, render_cards(recs),
                   gr.update(choices=story_choices(recs), value=None, visible=bool(recs)),
                   DETAIL_PLACEHOLDER, render_trace(response["trace"]),
                   render_metric_tiles(metrics, response["mock_llm"]), metrics_rows(metrics), idle_btn,
                   {"trace": response["trace"], "metrics": response["metrics"], "mock": response["mock_llm"],
                    "recs": {r["story_id"]: r for r in recs}})
    except Exception as e:
        yield (friendly_error("finding stories", e), no_change, no_change, no_change, no_change, no_change,
               no_change, idle_btn, no_change)


def on_select(story_id, search_state):
    no_change = gr.update()
    if not story_id or not search_state:
        # The radio also fires 'change' when a new search clears it; there is nothing to load then.
        yield no_change, no_change, no_change, no_change
        return
    search_trace = search_state["trace"]
    title = search_state["recs"].get(story_id, {}).get("title", "this story")
    yield (render_detail_loading(title), render_trace(search_trace, DETAIL_AGENTS, "writer"), no_change, no_change)
    try:
        for kind, payload in _stream(lambda cb: orchestrator.get_story_details(story_id, on_step=cb)):
            if kind == "step":
                remaining, running = _pending(DETAIL_AGENTS, payload)
                if payload and payload[-1]["status"] == "cache_hit":
                    remaining = []  # a cached story skips the Writer and Validator entirely
                yield no_change, render_trace(search_trace + payload, remaining, running), no_change, no_change
                continue
            response = payload
            metrics = combine_metrics(search_state["metrics"], response["metrics"], mock=search_state["mock"])
            if response.get("error") or not response.get("story"):
                detail = render_banner("error", f"{esc(response.get('error') or 'No story was produced.')} "
                                                "Try another story.")
            else:
                detail = render_detail(response["story"])
            yield (detail, render_trace(search_trace + response["trace"]),
                   render_metric_tiles(metrics, search_state["mock"]), metrics_rows(metrics))
    except Exception as e:
        yield friendly_error("writing this story", e), no_change, no_change, no_change


# analytics_enabled=False: Gradio otherwise sends usage telemetry to its servers; this app needs no external calls
# beyond the RSS feeds, the article pages and the share link.
with gr.Blocks(title="Story Discovery", analytics_enabled=False) as demo:
    search_state = gr.State(None)
    gr.HTML(f"""<div class="sd-header"><h1>Story Discovery</h1>
<p>Tell us how you feel. Five AI agents find real uplifting news that fits, then retell it in their own words.</p>
<span class="sd-mode">{esc(_mode_label())}</span></div>""")

    with gr.Group():
        mood_box = gr.Textbox(label="How are you feeling?", lines=1, max_lines=3, autofocus=True,
                              placeholder="e.g. stressed about exams, want something calming")
    with gr.Row(elem_classes="sd-chips"):
        chip_buttons = [gr.Button(m, size="sm", variant="secondary", elem_classes="sd-chip", scale=0, min_width=0)
                        for m in UI_EXAMPLE_MOODS]
    with gr.Row(equal_height=True):
        source_dd = gr.Dropdown(SOURCE_CHOICES, value="Any", label="Preferred source (optional)", scale=2)
        go_btn = gr.Button("Tell Me a Story", variant="primary", size="lg", elem_id="sd-go", scale=3)

    with gr.Row(elem_classes="sd-split"):
        with gr.Column(scale=5, min_width=320):
            status_html = gr.HTML("")
            story_radio = gr.Radio(choices=[], label="Choose a story to read", visible=False, elem_id="sd-pick")
            cards_html = gr.HTML("")
        with gr.Column(scale=7, min_width=320):
            detail_html = gr.HTML(DETAIL_PLACEHOLDER)

    with gr.Accordion("Agent Trace", open=True):
        trace_html = gr.HTML(render_trace([]))
    with gr.Accordion("Metrics", open=True):
        tiles_html = gr.HTML("")
        metrics_df = gr.Dataframe(headers=METRIC_HEADERS, value=[], interactive=False, label="Per-agent breakdown")

    for chip, text in zip(chip_buttons, UI_EXAMPLE_MOODS):
        # Filled in the browser (js, no server round-trip): with a Python handler, a quick click on the main
        # button could reach the server before the textbox updated and search with an empty mood.
        chip.click(fn=None, inputs=None, outputs=mood_box, js=f"() => {json.dumps(text)}", queue=False)
    search_outputs = [status_html, cards_html, story_radio, detail_html, trace_html, tiles_html, metrics_df,
                      go_btn, search_state]
    go_btn.click(on_search, inputs=[mood_box, source_dd], outputs=search_outputs)
    mood_box.submit(on_search, inputs=[mood_box, source_dd], outputs=search_outputs)
    story_radio.change(on_select, inputs=[story_radio, search_state],
                       outputs=[detail_html, trace_html, tiles_html, metrics_df])

print("UI built. Launch it with the next cell.")

# %% [markdown]
# ## 19. Launch the app
#
# **How to use it**
#
# 1. Run this cell. With **Runtime → Run all** it runs automatically.
# 2. Open the public link it prints (`https://….gradio.live`) in any browser
#    tab, or use the app that appears inline below this cell.
# 3. Type how you feel, or tap an example mood. Optionally pick a source,
#    then press **Tell Me a Story**.
# 4. Pick one of the stories to load its paraphrase and summary. The first
#    open of a story runs the Writer (roughly tens of seconds on a T4);
#    opening it again is instant.
# 5. Open **Agent Trace** to watch each agent step, and **Metrics** for
#    tokens, timing and GPU memory.
#
# **Notes**
#
# - The share link is temporary and only works while this Colab runtime is
#   running.
# - To stop the app, run `demo.close()` in a new cell, or use **Runtime →
#   Interrupt execution**. To free the GPU too, use **Runtime → Disconnect
#   and delete runtime**.
# - Re-running this cell closes the previous app first, so you never end up
#   with two copies running.
# - Without a GPU the app runs in mock mode (marked in the header): the whole
#   UI works with canned model outputs.

# %%
if "launched_demo" in globals():
    try:
        launched_demo.close()  # re-running this cell must not leave the previous app running on the same port
    except Exception:
        pass

print("=" * 72)
print("Story Discovery is starting.")
print(f"Mode: {_mode_label()}")
print("• Open the public https://….gradio.live link printed below (it works from any browser),")
print("  or use the app shown inline under this cell.")
print("• The link stays up only while this Colab runtime is running.")
print("• To stop: run  demo.close()  in a new cell, or Runtime → Interrupt execution.")
print("=" * 72)

launched_demo = demo
demo.queue(default_concurrency_limit=1).launch(share=True, theme=UI_THEME, css=UI_CSS, show_error=True)

# %% [markdown]
# ## 20. References
#
# This lists only what this notebook actually uses. All URLs were checked on
# 2026-10-12.
#
# **Models** (Hugging Face model cards; all ungated)
#
# | Agent | Model | License | Model card |
# |---|---|---|---|
# | Mood/Intent | Qwen2.5-1.5B-Instruct (Qwen team, Alibaba Cloud) | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
# | Classifier/Ranker | Phi-3.5-mini-instruct (Microsoft) | MIT | https://huggingface.co/microsoft/Phi-3.5-mini-instruct |
# | Writer | Qwen2.5-7B-Instruct (Qwen team, Alibaba Cloud) | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
#
# **Libraries** (versions as pinned in section 1; `torch` and `requests`
# are Colab's preinstalled builds)
#
# | Library | Used for | URL |
# |---|---|---|
# | PyTorch (`torch`) | tensors, CUDA, GPU memory stats | https://pytorch.org |
# | Hugging Face Transformers 5.17.0 | model/tokenizer loading, chat templates, generation | https://github.com/huggingface/transformers · docs: https://huggingface.co/docs/transformers |
# | Hugging Face Accelerate 1.15.0 | device placement (`device_map`) when loading | https://github.com/huggingface/accelerate |
# | bitsandbytes 0.50.2 | 4-bit NF4 quantization | https://github.com/bitsandbytes-foundation/bitsandbytes · docs: https://huggingface.co/docs/bitsandbytes/main |
# | Gradio 6.28.0 | web UI | https://www.gradio.app · https://github.com/gradio-app/gradio |
# | `allenai/gradio-theme` (Ai2) | Gradio theme loaded from the Hub (`gr.Theme.from_hub`) | https://huggingface.co/spaces/allenai/gradio-theme |
# | feedparser 6.0.14 | parsing RSS feeds | https://github.com/kurtmckee/feedparser |
# | trafilatura 2.2.0 | extracting article text and dates from web pages | https://github.com/adbar/trafilatura · docs: https://trafilatura.readthedocs.io |
# | Requests (Colab's preinstalled version) | HTTP fetching with timeouts | https://requests.readthedocs.io |
# | urllib3 (dependency of Requests) | `Retry` policy for HTTP retries/backoff | https://urllib3.readthedocs.io |
#
# The Python standard library (`json`, `re`, `difflib`, `threading`,
# `concurrent.futures`, `dataclasses`, `hashlib` and others) is used
# throughout but not listed separately.
#
# **Data sources**
#
# No datasets are used. Stories are fetched live, at run time, from these
# public RSS feeds:
#
# | Source | Site | Feed used |
# |---|---|---|
# | Good News Network | https://www.goodnewsnetwork.org | https://www.goodnewsnetwork.org/feed/ |
# | Positive News | https://www.positive.news | https://www.positive.news/feed/ |
# | Reasons to be Cheerful | https://reasonstobecheerful.world | https://reasonstobecheerful.world/feed/ |
# | The Optimist Daily | https://www.optimistdaily.com | https://www.optimistdaily.com/feed/ |
# | Sunny Skyz | https://www.sunnyskyz.com | https://www.sunnyskyz.com/rss_tebow.php |
#
# - **Copyright.** Article text stays with its publishers. The app shows
#   only model-written paraphrases and summaries, always with the source name
#   and a link to the original.
# - **Fallback stories.** The 8 bundled stories (section 7) are original
#   placeholder text written for this notebook, not copied from any source.
#   Their URLs use the reserved `example.org` domain, per RFC 2606:
#   https://www.rfc-editor.org/rfc/rfc2606
#
# **Platform:** Google Colab (https://colab.research.google.com). Its
# Secrets feature (`google.colab.userdata`) is read only if a gated model is
# configured, which it isn't by default.
#
# **Starter code:** none. All code in this notebook was written for this
# project; see the AI Assistance Declaration.
#
# **Development and testing only** (not used when the notebook runs):
# - Jupytext (https://github.com/jupytext/jupytext): this notebook is authored
#   as a percent-format `.py` file and converted to `.ipynb`.
# - Qwen2.5-0.5B-Instruct (https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct):
#   a small stand-in for local CPU tests of the real-model code path.
# - Playwright for Python (https://playwright.dev/python/): headless-browser
#   tests of the Gradio UI.
# - Claude Code (https://www.anthropic.com/claude-code): see the
#   declaration below.

# %% [markdown]
# ## 21. AI Assistance Declaration
#
# **Tool.** Claude Code, Anthropic's coding assistant
# (https://www.anthropic.com/claude-code), used in a local VS Code session.
# The first prompt ran on Claude Sonnet 5.5; all later prompts ran on Claude
# Opus 5.5.
#
# **What it was used for.** Under my direction and through the prompts
# below, Claude Code:
# - wrote the project rules and the architecture design
# - wrote essentially all of this notebook's code and markdown explanations
# - ran the local tests described below
#
# I chose the requirements, reviewed the design and code at each step, and
# made the decisions it raised: `[FILL: list your main decisions or changes,
# e.g. which models, kept the two-call Writer, etc.]`.
#
# **No AI service is used at run time.** The notebook calls only the three
# open-weight Hugging Face models listed in References. It makes no calls to
# Anthropic, OpenAI or any other proprietary API.
#
# **Major prompts** (summarized, not verbatim; prompts 1–12 are from
# `prompts_log.md`, kept during the main build)
#
# | # | Prompt (summary) | Result |
# |---|---|---|
# | 1 | Write a CLAUDE.md capturing the constraints (≥3 open-weight HF LLMs, no proprietary APIs, free Colab T4, Run all, required markdown cells) and rules: jupytext `notebook.py` + build script, `# UNVERIFIED:` labels, no hardcoded keys, keep a prompts log. No code. | Project rules file |
# | 2 | Design the architecture before coding: 5 agents (Mood/Intent small LLM, Retrieval tool, Classifier/Ranker mid LLM, Writer large LLM, Python Validator with retries), three T4-compatible models with license/VRAM and load strategy, dataclass schemas, categories, `USE_MOCK_LLM`, text flow diagram. | Architecture document |
# | 3 | Create the notebook's first cells: title/overview/how-to-run, pinned install cell, config cell (models, categories, word bounds, RSS sources, mock auto-detect, Colab Secrets only for gated models), GPU check; a markdown cell above every code cell; prefer ungated models. | Sections 1–3; model IDs, licenses and gating checked on the HF API; feeds fetched; pins checked on PyPI |
# | 4 | Add a ModelManager: load/unload the LLMs, 4-bit for the large one, `generate()` via chat templates returning text, real-tokenizer token counts, time, tokens/s, peak GPU memory; a mock backend with schema-valid outputs; cache clearing between swaps; model-choice explanation; smoke test. | Sections 4–5 |
# | 5 | Implement the Retrieval agent: feeds in config + startup health check, trafilatura extraction (title, source, URL, date), URL checks with timeouts and a polite User-Agent, dedupe, short-article filter, optional preferred source, retries and per-feed graceful failure, ~8 original placeholder fallback stories when fewer than 5 usable. | Sections 6–8 |
# | 6 | Implement the Mood/Intent agent (JSON intent, strict prompting, robust parser) and the Classifier/Ranker (allowed category, uplifting check, 0–10 mood fit, reason; batched and truncated input; top 5+ ranking); prompt templates as constants; prompt-design explanation; 3-mood test; invalid output must never crash. | Sections 9–12 |
# | 7 | Implement the Writer (shorter paraphrase in its own words + 70–90 word summary) and the Validator (deterministic word count, up to 3 re-prompts with the exact count, then programmatic trim/extend flagged in metadata, n-gram copy check); return a StoryDetail with per-call metrics; test cell. | Sections 13–14 |
# | 8 | Implement the Orchestrator: recommend (mood → retrieval → ranking, 5+ results) and lazy, cached `get_story_details`; an AgentTrace; per-agent and total metrics; JSON-serializable output; end-to-end tests (mock, and real when CUDA is present). | Section 15 |
# | 9 | Build the Gradio frontend: header, mood box with example chips, source dropdown, "Tell Me a Story" button, 5+ selectable story cards, detail panel, metrics panel, live Agent Trace, loading states, friendly errors, fallback note; launch with `share=True`; fully usable in mock mode. | Sections 17–19 |
# | 10 | Add an evaluation over 10 diverse moods (categories, % summaries within 70–90 words, retries, URL success, per-stage latency, tokens), a Results & Analysis template and an honest Known Limitations cell, behind a `RUN_EVAL` flag. | Section 16 |
# | 11 | Finish the documentation: overview with a diagram that renders in Colab, a markdown explanation for every code cell, Manual steps, References (only what is actually used, with URLs), and this declaration. | Overview, Manual steps, sections 20–21 |
# | 12 | Review the notebook as a skeptical grader (requirements → cells, Run All risks on a fresh T4, proprietary APIs, secrets, copied text, markdown coverage, open UNVERIFIED items), then fix what was found. | Build script with checks; mock fallback if real models fail to load; at least 5 recommendations guaranteed; `requests` no longer reinstalled; Gradio analytics off |
# | 13 | Add a hallucination check to the Validator (plain Python, no new model): pull numbers (percentages, money, years) and multi-word capitalized names out of the paraphrase and summary, and look for each in the source after normalizing number formats ("1,200" = "1200", "50%" = "50 percent", spelled-out numbers up to twenty). Report the facts not found and the share found for each text; warn (never block or regenerate) on any missing number or more than 2 missing names; show it next to the copy check in the detail view, HTML-escaped; add it to the evaluation table; add test cases (clean text, invented number, invented name, number-format variants) run in mock mode; keep false positives low; document the limits. | Fact check in section 13 with tests; fact-check tag in the story detail; new column and summary line in the evaluation |
#
# **How the output was reviewed and tested**
#
# *During development* (by Claude Code at my request, on a local Mac:
# CPU only, 8 GB RAM, with the pinned library versions installed):
#
# - **Facts were checked, not assumed:**
#   - model IDs, licenses and gating via the Hugging Face API
#   - package versions via PyPI
#   - every feed URL by fetching it
#   - library APIs by inspecting the pinned versions rather than relying on
#     memory (Transformers 5.x and Gradio 6.x differ from earlier versions)
#
#   - that the pinned packages install together on Colab's platform (a pip
#     dry run for Linux x86-64 / Python 3.12)
#   - that Qwen2.5-1.5B and Phi-3.5-mini don't overflow in fp16, which the
#     T4 requires (one fp16 forward pass each on CPU)
#
#   Anything that could not be checked locally is marked `UNVERIFIED` in the
#   code and text.
# - **Execution.** The notebook was converted with jupytext and every code
#   cell was executed in order:
#   - in **mock mode**
#   - on the **real model code path**, with Qwen2.5-0.5B-Instruct standing in
#     for all three models. The real models don't fit in 8 GB of RAM, so
#     this tests plumbing, not output quality.
# - **Targeted tests:**
#   - JSON-parser robustness against fenced, chatty, single-quoted, truncated
#     and garbage replies
#   - Validator cases: trim, extend, copy check, fact check
#   - batched and single generation giving identical output
#   - 7 orchestrator checks, including JSON serializability and the cache hit
#   - simulated agent crashes
#   - the Gradio UI in a headless browser: light and dark mode, a 390 px
#     phone layout, empty input, two tabs, and HTML-escaping of hostile
#     titles
# - **Problems found by this testing and fixed:**
#   - the classifier copied the reader's preferred category onto unrelated
#     stories
#   - a stray title line in Writer output, and repeated sentences when a
#     summary was extended
#   - a malformed HTML entity in one feed
#   - a race between the example chips and the search button
#   - horizontal scrolling on phones
#
# *Not tested during development:* the notebook on an actual Colab T4 GPU,
# including 4-bit loading, the out-of-memory fallback, real-model output
# quality and latency, and the public `share=True` link.
#
# *My own review and T4 run:* `[FILL: what you personally read, changed and
# verified; the date and outcome of your Colab T4 "Run all"]`.
