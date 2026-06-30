#!/usr/bin/env python3
"""Serving/prompting configuration governs measured protocol fidelity.

Three conditions on the SAME task/seeds:
  native      : harness default, sends OpenAI `tools=`. On Ollama this is gated
                per model (qwen: accepted, emits text; llama3.2: native tool_calls;
                phi3/gemma: request REJECTED with HTTP 400 -> model never runs).
  native+hint : `tools=` still sent, PLUS the plain-text tool list + call protocol
                injected into the prompt (LOCA_TOOL_HINT=1). Isolates the prompt
                guidance from the serving channel. (Only for models the server
                accepts: qwen family + llama3.2.)
  text-tools  : `tools=` dropped, tool list + protocol in the prompt, calls parsed
                from text (LOCA_TEXT_TOOLS=1). Uniform across all models.

We report per-seed fidelity (each of 8 episodes weighted equally; avoids one long
looping episode dominating the pooled rate) with the standard error across seeds.
Writes serving_modes.csv + serving_modes.png.
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_protocol import classify_dir, OUTS

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = ["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b", "qwen2.5-coder:3b",
          "qwen2.5-coder:7b", "qwen2.5-coder:14b",
          "llama3.2:latest", "phi3:latest", "gemma3:4b", "gemma3:270m",
          "deepseek-v4-flash"]


def stats(tag, model):
    """Return (seed_mean, se, pooled, n_turns, n_rejected) for a condition, or None."""
    pat = os.path.join(OUTS, f"inf_react_agg_grad_{tag}{model}_*")
    cs = sorted(glob.glob(pat), key=os.path.getmtime)
    if not cs:
        return None
    r = classify_dir(cs[-1])
    if not r:
        return None
    ps = r["per_seed"]
    fids = [v / d for v, d in ps]
    if fids:
        m = sum(fids) / len(fids)
        var = sum((f - m) ** 2 for f in fids) / len(fids)
        se = (var / len(fids)) ** 0.5
    else:
        m, se = 0.0, 0.0
    pooled = (r["counts"].get("valid_call", 0) / r["n_turns"]) if r["n_turns"] else 0.0
    return (m, se, pooled, r["n_turns"], r["counts"].get("rejected", 0))


def main():
    rows = []
    for m in MODELS:
        rec = {"model": m}
        for cond, tag in (("native", ""), ("native_hint", "hint_"),
                          ("text_tools", "tt_"), ("constrained", "con_")):
            s = stats(tag, m)
            if s is None:
                rec[cond + "_seed"] = rec[cond + "_se"] = rec[cond + "_pooled"] = None
                rec[cond + "_n"] = rec[cond + "_rej"] = None
            else:
                rec[cond + "_seed"], rec[cond + "_se"], rec[cond + "_pooled"], \
                    rec[cond + "_n"], rec[cond + "_rej"] = (round(s[0], 3), round(s[1], 3),
                                                           round(s[2], 3), s[3], s[4])
        rows.append(rec)

    with open(os.path.join(HERE, "serving_modes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"{'model':20s} {'native':20s} {'+hint':20s} {'text-tools':20s} {'constrained':20s}")
    for r in rows:
        def f(c):
            if r[c + "_seed"] is None:
                return "--"
            return f"{r[c+'_seed']*100:3.0f}/{r[c+'_pooled']*100:3.0f}/n{r[c+'_n']}" + (f" rej{r[c+'_rej']}" if r[c+'_rej'] else "")
        print(f"{r['model']:20s} {f('native'):20s} {f('native_hint'):20s} {f('text_tools'):20s} {f('constrained'):20s}")

    # ---- grouped bar: per-seed fidelity, 3 conditions ----
    labels = [r["model"].replace("qwen2.5-coder:", "qwen-").replace(":latest", "")
              .replace("gemma3:", "gemma-").replace("deepseek-v4-flash", "deepseek\n(cloud)")
              for r in rows]
    x = range(len(labels))
    w = 0.27
    conds = [("native", "#9aa0a6", -w), ("native_hint", "#4c78c8", 0.0),
             ("text_tools", "#2a7", w)]
    nice = {"native": "native FC (default)", "native_hint": "native + text hint",
            "text_tools": "text-tools (uniform)"}
    fig, ax = plt.subplots(figsize=(11, 5))
    for cond, color, off in conds:
        xs = [i + off for i in x]
        ys = [(r[cond + "_seed"] or 0.0) for r in rows]
        es = [(r[cond + "_se"] or 0.0) for r in rows]
        present = [r[cond + "_seed"] is not None for r in rows]
        ax.bar([xi for xi, p in zip(xs, present) if p],
               [yi for yi, p in zip(ys, present) if p], w,
               yerr=[ei for ei, p in zip(es, present) if p],
               capsize=2, label=nice[cond], color=color, error_kw={"elinewidth": 0.8})
    # mark native rejections (at the native bar position)
    for i, r in enumerate(rows):
        if r["native_rej"]:
            ax.text(i - w, 0.02, "rej", ha="center", va="bottom", fontsize=7,
                    color="#c0392b", rotation=90)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("per-seed protocol fidelity\n(valid in-schema rate, mean ± SE over 8 seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Serving / prompting configuration governs measured protocol fidelity\n"
                 "(\"rej\" = native request rejected by Ollama; model never ran)", fontsize=11)
    ax.axvline(4.5, color="0.85", lw=1, ls="--")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "serving_modes.png"), dpi=150)
    print("\nwrote serving_modes.png + serving_modes.csv")


if __name__ == "__main__":
    main()
