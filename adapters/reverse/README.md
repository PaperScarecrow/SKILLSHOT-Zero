---
base_model: unsloth/gemma-3-270m-it
library_name: peft
pipeline_tag: text-generation
tags:
- skillshot
- lora
- tool
- procedural-skill
- runtime-forged
---

# SKILLSHOT tool — `reverse`

A hot-swappable LoRA **tool** for the SKILLSHOT tool-changer. It teaches the frozen
`gemma-3-270m-it` "machine" to reverse a sequence of digits. This is the **runtime-forged**
example: in the tool-changer POC, `reverse` arrives as an *unseen* request with no matching
tool in the magazine; the system forges a new tool on the fly, clears a held-out test, and
**memorializes** it into the magazine (project → test → memorialize).

## Task

Reverse the order of the input digits.

```
Prompt:  Reverse the order of these digits.
         Output only the reversed digits separated by single spaces.
         Input: 4 1 7 2
Output:  2 7 1 4
```

Inputs are random 4–7 digit sequences; the test set is disjoint from training.

## Results (CPU, gemma-3-270m-it)

In the tool-changer loop (`proof/tool_changer_results.json`):

- Arrived as a **miss** (no routable tool in the magazine `[cipher, sort, addmod]`).
- Forged at runtime, passed the held-out gate (**gate_acc 1.000**), and was **memorialized**
  → magazine becomes `[cipher, sort, addmod, reverse]`.
- After admission, end-to-end answer accuracy across the magazine is **1.000**, hot-swap
  ~7.8 ms.

## Training

| | |
|---|---|
| Base model | `unsloth/gemma-3-270m-it` (frozen) |
| Method | LoRA (PEFT 0.18.1), rank **32**, alpha **64**, dropout 0 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | **7.59M (≈2.75%** of base) |
| Optimizer | AdamW, lr 2e-4 |
| Steps | 500 (runtime forge) |
| Data | synthetic, disjoint train/test |
| Precision | fp32 |

Reproduce: `cd proof && CUDA_VISIBLE_DEVICES="" python tool_changer_poc.py --steps 500`
(forges `reverse` as the miss skill).

## Load and use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "unsloth/gemma-3-270m-it"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "adapters/reverse", adapter_name="reverse")
```

See [`../README.md`](../README.md) for the full magazine and the tool-changer demo.
