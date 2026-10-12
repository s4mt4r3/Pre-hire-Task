"""Build notebook.ipynb from notebook.py (jupytext percent format) and run pre-submission checks.

Usage:
    pip install jupytext==1.19.6
    python build.py

Hard failures (exit code 1) block the build; warnings list what still needs a human.
"""
import re
import sys
from pathlib import Path

try:
    import jupytext
    import nbformat
except ImportError:
    sys.exit("jupytext is required: pip install jupytext==1.19.6")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "notebook.py"
OUT = ROOT / "notebook.ipynb"

PROPRIETARY_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(openai|anthropic|google\.generativeai|google\.genai|cohere|mistralai|groq)\b", re.M
)
SECRET = re.compile(r"hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}")


def main():
    nb = jupytext.read(SRC)
    cells = nb.cells
    errors, warnings = [], []

    for i, cell in enumerate(cells):
        if cell.cell_type == "code" and (i == 0 or cells[i - 1].cell_type != "markdown"):
            first = cell.source.splitlines()[0] if cell.source else ""
            errors.append(f"code cell {i} has no markdown cell above it: {first[:60]!r}")
        if cell.cell_type == "code" and PROPRIETARY_IMPORT.search(cell.source):
            errors.append(f"code cell {i} imports a proprietary LLM SDK")
        if SECRET.search(cell.source):
            errors.append(f"cell {i} contains something that looks like an API key or token")

    code_cells = [c for c in cells if c.cell_type == "code"]
    if not code_cells or "launch(share=True" not in code_cells[-1].source:
        errors.append("the final code cell does not launch the Gradio app with share=True")

    src_lines = SRC.read_text(encoding="utf-8").splitlines()
    for label, pattern in [("UNVERIFIED", "UNVERIFIED"), ("TODO(spec)", "TODO(spec)"), ("[FILL]", "[FILL")]:
        hits = [n for n, line in enumerate(src_lines, 1) if pattern in line]
        if hits:
            warnings.append(f"{len(hits)} line(s) with {label}: notebook.py lines {hits}")

    for e in errors:
        print(f"ERROR: {e}")
    if errors:
        sys.exit(f"Build failed with {len(errors)} error(s); {OUT.name} was not written.")

    # Outputs are never stored in notebook.py, so the built notebook ships clean.
    nbformat.validate(nb)
    jupytext.write(nb, OUT, fmt="ipynb")
    n_md = sum(c.cell_type == "markdown" for c in cells)
    print(f"Wrote {OUT.name}: {len(cells)} cells ({len(code_cells)} code, {n_md} markdown).")
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
