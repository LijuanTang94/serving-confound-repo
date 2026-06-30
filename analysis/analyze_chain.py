#!/usr/bin/env python3
"""Second-task replication: native vs text-tools on the dependency-chain task.

Checks whether the serving confound (rejection -> silent 0%) and the prompt-driven
recovery replicate on a different task (swe_chain_s2l), using the same per-turn
classifier and per-seed metric as the aggregation task.
Writes chain_compare.csv.
"""
import csv
import glob
import os

from analyze_protocol import classify_dir, OUTS

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b", "qwen2.5-coder:3b",
          "qwen2.5-coder:7b", "qwen2.5-coder:14b",
          "llama3.2:latest", "phi3:latest", "gemma3:4b", "gemma3:270m"]


def stat(tag, model):
    pat = os.path.join(OUTS, f"inf_react_chain_grad_{tag}{model}_*")
    cs = sorted(glob.glob(pat), key=os.path.getmtime)
    if not cs:
        return None
    r = classify_dir(cs[-1])
    if not r:
        return None
    pooled = (r["counts"].get("valid_call", 0) / r["n_turns"]) if r["n_turns"] else 0.0
    return (round(r["seed_mean"], 3), round(pooled, 3), r["n_turns"],
            r["counts"].get("rejected", 0))


def main():
    rows = []
    for m in MODELS:
        nat = stat("", m)       # native: inf_react_chain_grad_<model>
        tt = stat("tt_", m)     # text-tools: inf_react_chain_grad_tt_<model>
        rows.append({"model": m,
                     "native": nat, "text_tools": tt})
    print(f"{'model':20s} {'native seed/pool/n/rej':26s} {'text-tools seed/pool/n/rej':26s}")
    out = []
    for r in rows:
        def f(s):
            return "--" if s is None else f"{s[0]*100:3.0f}/{s[1]*100:3.0f}/n{s[2]} rej{s[3]}"
        print(f"{r['model']:20s} {f(r['native']):26s} {f(r['text_tools']):26s}")
        out.append({
            "model": r["model"],
            "native_seed": r["native"][0] if r["native"] else None,
            "native_pool": r["native"][1] if r["native"] else None,
            "native_n": r["native"][2] if r["native"] else None,
            "native_rej": r["native"][3] if r["native"] else None,
            "tt_seed": r["text_tools"][0] if r["text_tools"] else None,
            "tt_pool": r["text_tools"][1] if r["text_tools"] else None,
            "tt_n": r["text_tools"][2] if r["text_tools"] else None,
            "tt_rej": r["text_tools"][3] if r["text_tools"] else None,
        })
    with open(os.path.join(HERE, "chain_compare.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("\nwrote chain_compare.csv")


if __name__ == "__main__":
    main()
