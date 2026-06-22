---
base_model: unsloth/gemma-3-270m-it
library_name: peft
pipeline_tag: text-generation
tags:
- skillshot
- lora
- tool
- procedural-skill
---

# SKILLSHOT tool — `cipher`

A hot-swappable LoRA **tool** for the SKILLSHOT tool-changer. It teaches the frozen
`gemma-3-270m-it` "machine" a single procedural skill: apply a fixed secret digit
substitution to a sequence of digits. This is the headline skill-shot example — the base
model scores **0%** and the tool reaches **100% on held-out inputs**.

## Task

Apply a secret one-to-one digit permutation to each input digit, in order.

```
Prompt:  Apply the secret digit substitution to each digit, in order.
         Output only the resulting digits separated by single spaces.
         Input: 4 1 7 2
Output:  <perm[4]> <perm[1]> <perm[7]> <perm[2]>
```

The permutation is fixed per training run and never given in the prompt — the tool has to
*encode the procedure in its weights*. Inputs are random 4–7 digit sequences; the test set
is disjoint from training.

## Results (CPU, gemma-3-270m-it)

| Metric | Base (no tool) | + `cipher` tool |
|---|---|---|
| Exact match | **0.000** | **1.000** |
| Char similarity | 0.414 | 1.000 |

Wrong-tool guard: loading the `sort` tool for a cipher request scores **0.000** — the tools
are specific, not generically helpful. Measured in `proof/skillshot_lift_cipher.json` and
`proof/tool_changer_results.json`.

## Training

| | |
|---|---|
| Base model | `unsloth/gemma-3-270m-it` (frozen) |
| Method | LoRA (PEFT 0.18.1), rank **32**, alpha **64**, dropout 0 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | **7.59M (≈2.75%** of base) |
| Optimizer | AdamW, lr 2e-4 |
| Steps | 600 |
| Data | 200 train / 80 held-out test, synthetic |
| Precision | fp32 |

Reproduce: `cd proof && CUDA_VISIBLE_DEVICES="" python gemma_skillshot.py --skill cipher --steps 600 --rank 32`

## Load and use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "unsloth/gemma-3-270m-it"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "adapters/cipher", adapter_name="cipher")
# model.set_adapter("cipher")  # then generate as usual
```

See [`../README.md`](../README.md) for the full magazine and the tool-changer demo.
