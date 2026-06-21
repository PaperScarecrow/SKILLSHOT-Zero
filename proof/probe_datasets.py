#!/usr/bin/env python
"""Probe HF datasets (network + schema) for the real-tool skills. Streaming = cheap."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "0"          # datasets need network (gemma stays offline elsewhere)
os.environ["TRANSFORMERS_OFFLINE"] = "0"
from datasets import load_dataset

CANDS = {
    "vuln:devign":   ("google/code_x_glue_cc_defect_detection", None, "test"),
    "vuln:diversevul": ("claudios/DiverseVul", None, "train"),
    "code:mbpp":     ("mbpp", None, "test"),
    "code:cruxeval": ("cruxeval-org/cruxeval", None, "test"),
    "code:humaneval": ("openai_humaneval", None, "test"),
    "cve:circl":     ("CIRCL/vulnerability", None, "train"),
    "cve:cvss":      ("nvd", None, "train"),
}

for name, (rid, cfg, split) in CANDS.items():
    try:
        ds = load_dataset(rid, cfg, split=split, streaming=True)
        ex = next(iter(ds))
        print(f"OK   {name:18} {rid} | fields={list(ex.keys())}")
        for k, v in ex.items():
            print(f"        {k}: {str(v)[:90]}")
    except Exception as e:
        print(f"FAIL {name:18} {rid} :: {repr(e)[:160]}")
