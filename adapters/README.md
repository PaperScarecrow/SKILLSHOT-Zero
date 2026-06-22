# Example tools (the magazine)

A curated set of **example LoRA tools** for the SKILLSHOT tool-changer — the "magazine" of
hot-swappable skill-adapters that snap onto the frozen `gemma-3-270m-it` "machine". Each tool
is a real trained adapter (not a stub); the weights ship via **Git LFS**.

The base model is **not** vendored — it loads from your local HF cache
(`unsloth/gemma-3-270m-it`, the non-gated mirror used throughout the POC).

## The four examples

| Tool | What it does | Data | Base → tool | Story |
|---|---|---|---|---|
| [`cipher`](cipher/) | apply a secret digit substitution | synthetic | **0.000 → 1.000** exact | headline skill-shot lift |
| [`reverse`](reverse/) | reverse a digit sequence | synthetic | forged → **1.000** | **runtime-forged & memorialized** |
| [`vuln-detect`](vuln-detect/) | C function → vulnerable/secure | Devign (HF) | 0.353 → **0.593** acc | real-data tool (appreciable lift) |
| [`cve-severity`](cve-severity/) | CVE desc → CVSS severity | CVE/CWE (HF) | F1 0.185 → **0.465** | real-data tool (macro-F1 lift) |

These four cover every claim in the top-level README: the headline 0→1 procedural lift, the
project→test→memorialize forge loop, and the two real public-dataset tools. The full set of
eight adapters (also `sort`, `addmod`, `code-match`, `code-output`) is reproducible from the
`proof/` scripts.

## Common spec

All four are LoRA adapters over `gemma-3-270m-it`: rank **32**, alpha **64**, targeting the
attention + MLP projections (`q,k,v,o,gate,up,down`), **7.59M trainable params (≈2.75%** of
base), fp32. Per-tool training details and reproduce commands are in each tool's `README.md`.

## Getting the weights (Git LFS)

```bash
git lfs install        # once per machine
git clone <repo>       # or: git lfs pull   in an existing clone
```

Without LFS you'll get small pointer files instead of the `.safetensors` weights.

## Load a tool

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "unsloth/gemma-3-270m-it"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)

# snap a tool onto the machine
model = PeftModel.from_pretrained(model, "adapters/cipher", adapter_name="cipher")

# hot-swap more tools into the same model, then pick one
model.load_adapter("adapters/reverse", adapter_name="reverse")
model.set_adapter("reverse")
```

For the full route → swap → (miss) → forge → memorialize loop, see
[`../proof/tool_changer_poc.py`](../proof/tool_changer_poc.py) and
[`../proof/PROOF_RESULTS.md`](../proof/PROOF_RESULTS.md).
