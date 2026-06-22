---
base_model: unsloth/gemma-3-270m-it
library_name: peft
pipeline_tag: text-generation
tags:
- skillshot
- lora
- tool
- security
- vulnerability-detection
---

# SKILLSHOT tool — `vuln-detect`

A hot-swappable LoRA **tool** for the SKILLSHOT tool-changer, forged from a **real public
dataset**: binary vulnerability detection on C functions. This is one of the "real tools"
examples showing the same skill-shot mechanism produces measurable lift on genuine data, not
just synthetic procedures.

## Task

Classify a C function as `vulnerable` or `secure`.

```
Prompt:  Classify the following C function as 'vulnerable' or 'secure'.
         Answer with exactly one word.
         ```c
         <C function>
         ```
         Answer:
Output:  vulnerable | secure
```

**Dataset:** Devign — `google/code_x_glue_cc_defect_detection` (downloaded from the HF Hub on
first run; not vendored here).

## Results (CPU, gemma-3-270m-it)

| Metric | Base (no tool) | + `vuln-detect` tool |
|---|---|---|
| Accuracy | 0.353 | **0.593** |
| Lift | — | **+0.240** |
| Majority-class baseline | 0.560 | — |

Verdict in `proof/real_tools_results.json`: **APPRECIABLE LIFT** (the tool beats both the
zero-shot base and the majority-class baseline). n_train 300 / n_test 150.

> Honest boundary: a 270M base is small; this tool demonstrates *direction and mechanism* on
> real data, not production-grade detection.

## Training

| | |
|---|---|
| Base model | `unsloth/gemma-3-270m-it` (frozen) |
| Method | LoRA (PEFT 0.18.1), rank **32**, alpha **64**, dropout 0 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | **7.59M (≈2.75%** of base) |
| Optimizer | AdamW, lr 2e-4 |
| Steps | 300 |
| Data | 300 train / 150 test (Devign) |
| Precision | fp32 |

Reproduce: `cd proof && CUDA_VISIBLE_DEVICES="" python real_tools_poc.py --steps 300`

## Load and use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "unsloth/gemma-3-270m-it"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "adapters/vuln-detect", adapter_name="vuln-detect")
```

See [`../README.md`](../README.md) for the full magazine and the tool-changer demo.
