#!/usr/bin/env python3
"""Seed-level bootstrap 95% CIs for per-seed protocol fidelity.

Proportions on 4-8 seeds are non-normal, so we replace the mean +/- SE summary
with a seed-level bootstrap: resample the per-seed fidelities with replacement
(10k draws), recompute the seed mean each time, and take the 2.5/97.5 percentiles.
This reuses the exact per-turn classifier (classify_dir) so the point estimates
match Table 1; it only adds an honest interval that is wide where n is thin.

Reads the same raw trajectory outputs as analyze_serving.py (set LOCA_OUTPUTS).
Writes serving_modes.csv (now with *_lo / *_hi columns) and regenerates
serving_modes.png with bootstrap 95% CI error bars.
"""
import csv, glob, os, random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_protocol import classify_dir, OUTS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")
B = 10000
SEED = 20260630

MODELS = ["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b", "qwen2.5-coder:3b",
          "qwen2.5-coder:7b", "qwen2.5-coder:14b",
          "llama3.2:latest", "phi3:latest", "gemma3:4b", "gemma3:270m",
          "deepseek-v4-flash"]
CONDS = [("native", ""), ("native_hint", "hint_"), ("text_tools", "tt_")]


def boot_ci(props, rng):
    """Percentile bootstrap 95% CI of the mean of per-seed fidelities."""
    n = len(props)
    if n == 0:
        return None, None, None
    mean = sum(props) / n
    if n == 1:
        return mean, mean, mean          # single seed: interval is the point
    draws = sorted(sum(props[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(B))
    return mean, draws[int(0.025 * B)], draws[int(0.975 * B)]


def cond_stats(tag, model, rng):
    cs = sorted(glob.glob(os.path.join(OUTS, f"inf_react_agg_grad_{tag}{model}_*")),
                key=os.path.getmtime)
    if not cs:
        return None
    r = classify_dir(cs[-1])
    if not r:
        return None
    props = [v / d for v, d in r["per_seed"] if d > 0]
    mean, lo, hi = boot_ci(props, rng)
    pooled = (r["counts"].get("valid_call", 0) / r["n_turns"]) if r["n_turns"] else 0.0
    return dict(seed=mean, lo=lo, hi=hi, pooled=pooled, n=r["n_turns"],
                rej=r["counts"].get("rejected", 0), n_seeds=len(props))


def main():
    rng = random.Random(SEED)
    rows = []
    for m in MODELS:
        rec = {"model": m}
        for cond, tag in CONDS:
            s = cond_stats(tag, m, rng)
            for k in ("seed", "lo", "hi", "pooled", "n", "rej", "n_seeds"):
                rec[f"{cond}_{k}"] = (None if s is None else
                                      (round(s[k], 3) if isinstance(s[k], float) else s[k]))
        rows.append(rec)

    cols = ["model"] + [f"{c}_{k}" for c, _ in CONDS
                        for k in ("seed", "lo", "hi", "pooled", "n", "rej", "n_seeds")]
    with open(os.path.join(RESULTS, "serving_modes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print(f"{'model':22s} {'native seed[95% CI]':22s} {'+hint':22s} {'text-tools':22s}")
    for r in rows:
        def f(c):
            if r[f"{c}_seed"] is None:
                return "--"
            if r[f"{c}_rej"] and r[f"{c}_n"] == 0:
                return "rej"
            return (f"{r[c+'_seed']*100:3.0f}[{r[c+'_lo']*100:3.0f},{r[c+'_hi']*100:3.0f}]"
                    f" n{r[c+'_n']}")
        print(f"{r['model']:22s} {f('native'):22s} {f('native_hint'):22s} {f('text_tools'):22s}")

    # ---- regenerate figure with bootstrap 95% CI error bars ----
    labels = [r["model"].replace("qwen2.5-coder:", "qwen-").replace(":latest", "")
              .replace("gemma3:", "gemma-").replace("deepseek-v4-flash", "deepseek\n(cloud)")
              for r in rows]
    x = range(len(labels)); w = 0.27
    conds = [("native", "#9aa0a6", -w), ("native_hint", "#4c78c8", 0.0),
             ("text_tools", "#2a7", w)]
    nice = {"native": "native FC (default)", "native_hint": "native + text hint",
            "text_tools": "text-tools (uniform)"}
    fig, ax = plt.subplots(figsize=(11, 5))
    for cond, color, off in conds:
        xs, ys, lo_err, hi_err = [], [], [], []
        for i, r in enumerate(rows):
            if r[f"{cond}_seed"] is None or (r[f"{cond}_rej"] and r[f"{cond}_n"] == 0):
                continue
            xs.append(i + off); ys.append(r[f"{cond}_seed"])
            lo_err.append(r[f"{cond}_seed"] - r[f"{cond}_lo"])
            hi_err.append(r[f"{cond}_hi"] - r[f"{cond}_seed"])
        ax.bar(xs, ys, w, yerr=[lo_err, hi_err], capsize=2, label=nice[cond],
               color=color, error_kw={"elinewidth": 0.8})
    for i, r in enumerate(rows):
        if r["native_rej"] and r["native_n"] == 0:
            ax.text(i - w, 0.02, "rej", ha="center", va="bottom", fontsize=7,
                    color="#c0392b", rotation=90)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("per-seed protocol fidelity\n(valid in-schema rate, bootstrap 95% CI)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Serving / prompting configuration governs measured protocol fidelity\n"
                 "(error bars: seed-level bootstrap 95% CI, 10k resamples; "
                 "\"rej\" = native request rejected, model never ran)", fontsize=10)
    ax.axvline(4.5, color="0.85", lw=1, ls="--")
    ax.legend(loc="upper right", fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES, "serving_modes.png"), dpi=150)
    print("\nwrote serving_modes.png + serving_modes.csv (with bootstrap 95% CIs)")


if __name__ == "__main__":
    main()
