#!/usr/bin/env python3
"""Pilot analyzer for Direction A: tool-call PROTOCOL fidelity vs model capability.

For each model's run (one output dir per model), classify every assistant turn's
tool-call behavior and count hallucinated-tool-name events, then plot the
composition vs model (ordered by rough capability).

Per assistant message classification:
  native        : native tool_calls, content has no call-like text
  text_format   : tool_calls present BUT content contains a call-like block
                  (our LOCA patch parsed a TEXT tool call into tool_calls)
  text_unparsed : no tool_calls but content looks like a call (protocol failure)
  prose_no_call : non-empty content, no tool_calls, no call-like text
Hallucinated names: tool-result messages saying "not found".
"""
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Point this at your LOCA-bench run outputs (override with env LOCA_OUTPUTS).
OUTS = os.environ.get("LOCA_OUTPUTS", os.path.join(HERE, "..", "outputs"))

# All models under the serving-uniform text-tools protocol. Qwen family by scale,
# then non-qwen (different families / tool-training), then the cloud anchor. The
# composition shows the qwen family is uniformly valid (no threshold) while the
# non-qwen models fail in different ways (phi3 prose, gemma-270m hallucination).
ORDER = ["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b", "qwen2.5-coder:3b",
         "qwen2.5-coder:7b", "qwen2.5-coder:14b",
         "llama3.2:latest", "phi3:latest", "gemma3:4b", "gemma3:270m",
         "deepseek-v4-flash"]
# Read the serving-uniform "text-tools" runs (LOCA_TEXT_TOOLS=1): tools described
# in-prompt as text, no native `tools=` param, calls parsed from text for EVERY
# model. This removes the Ollama confound where some models accept `tools=`
# (native/text) and others (phi3, gemma3) reject the request outright.
RUN_PREFIX = "inf_react_agg_grad_tt_"

CALL_LIKE = re.compile(r"```|\"name\"\s*:|\bfunction\b|\btool_call\b", re.I)
NOTFOUND = re.compile(r"not found|no such tool|unknown tool", re.I)
# Serving-layer rejection: the harness writes this when the request was refused
# (e.g. Ollama HTTP 400 "does not support tools"). The model never produced output,
# so this is NOT a protocol turn and is excluded from the fidelity denominator.
REJECTED = re.compile(r"Failed to get response", re.I)


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - half), min(1.0, centre + half))


def find_msg_lists(o):
    out = []
    def rec(x):
        if isinstance(x, list) and x and isinstance(x[0], dict) and "role" in x[0]:
            out.append(x); return
        if isinstance(x, dict):
            for v in x.values(): rec(v)
        elif isinstance(x, list):
            for v in x: rec(v)
    rec(o)
    return out


def classify_dir(d):
    # read the canonical per-state trajectory.json (one messages list per state);
    # do NOT recurse all_trajectories.json (it nests/duplicates -> overcounts).
    traj_files = glob.glob(os.path.join(d, "tasks", "*", "state*", "trajectory.json"))
    if not traj_files:
        return None
    counts = defaultdict(int)
    n_native = 0
    n_turns = 0
    per_seed = []          # (valid, denom) per state, denom excludes rejected turns

    def text(m):
        c = m.get("content", "")
        return c if isinstance(c, str) else json.dumps(c)

    for tf in traj_files:
        try:
            msgs = json.load(open(tf)).get("messages", [])
        except Exception:
            continue
        seed_valid = seed_denom = 0
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            content = text(m)
            # serving-layer rejection: not a protocol turn, excluded from denominator
            if REJECTED.search(content or ""):
                counts["rejected"] += 1
                continue
            tc = bool(m.get("tool_calls"))
            call_like = bool(CALL_LIKE.search(content or ""))
            parseable = tc  # our patch turns a recognized text call into tool_calls
            native = tc and not call_like
            if parseable:
                # look ahead to the next tool result to see if the tool existed
                notfound = False
                for j in range(i + 1, min(i + 4, len(msgs))):
                    if msgs[j].get("role") in ("tool", "function"):
                        notfound = bool(NOTFOUND.search(text(msgs[j]) or ""))
                        break
                if notfound:
                    counts["halluc_call"] += 1            # parseable but tool not in schema
                else:
                    counts["valid_call"] += 1             # parseable + in-schema
                    seed_valid += 1
                    if native:
                        n_native += 1
            elif call_like:
                counts["unparseable"] += 1                # looks like a call, no parser reads it
            elif (content or "").strip():
                counts["prose"] += 1                      # talks, never attempts a call
            else:
                continue
            n_turns += 1
            seed_denom += 1
        if seed_denom > 0:
            per_seed.append((seed_valid, seed_denom))
    lo, hi = wilson(counts.get("valid_call", 0), n_turns)
    # per-seed mean fidelity (each seed weighted equally; avoids one long episode
    # dominating the pooled rate) + its spread
    seed_fids = [v / d for v, d in per_seed]
    seed_mean = sum(seed_fids) / len(seed_fids) if seed_fids else 0.0
    return {"counts": dict(counts), "n_native": n_native, "n_turns": n_turns,
            "per_seed": per_seed, "ci": (lo, hi),
            "seed_mean": seed_mean, "n_seeds": len(per_seed)}


def main():
    rows = {}
    for label in ORDER:
        cands = sorted(glob.glob(os.path.join(OUTS, f"{RUN_PREFIX}*{label}*")),
                       key=os.path.getmtime)
        if not cands:
            print(f"[skip] no run for {label}")
            continue
        r = classify_dir(cands[-1])
        if r:
            rows[label] = r

    print(f"\n{'model':22s} {'turns':>6s} {'valid':>6s} {'halluc':>6s} {'unparse':>7s} {'prose':>6s} {'rej':>5s}  fidelity")
    cats = ["valid_call", "halluc_call", "unparseable", "prose", "rejected"]
    labels, fracs = [], {c: [] for c in cats}
    for label in ORDER:
        if label not in rows:
            continue
        c = rows[label]["counts"]
        # composition denominator includes non-response (rejected) turns, to show
        # the full per-turn behavior; fidelity (reported elsewhere) excludes them.
        n = max(1, rows[label]["n_turns"] + c.get("rejected", 0))
        labels.append(label.replace("qwen2.5-coder", "qwen"))
        for cat in cats:
            fracs[cat].append(c.get(cat, 0) / n)
        fid = c.get("valid_call", 0) / max(1, rows[label]["n_turns"])
        print(f"{label:22s} {rows[label]['n_turns']:>6d} "
              f"{c.get('valid_call',0):>6d} {c.get('halluc_call',0):>6d} "
              f"{c.get('unparseable',0):>7d} {c.get('prose',0):>6d} {c.get('rejected',0):>5d}  {fid:.2f}")

    if not labels:
        print("no data to plot"); return
    # stacked bar: per-turn protocol-outcome composition
    plt.figure(figsize=(7.8, 4.8))
    bottom = [0] * len(labels)
    colors = {"valid_call": "#2a7", "halluc_call": "#fb3", "unparseable": "#e74",
              "prose": "#a33", "rejected": "#888"}
    nice = {"valid_call": "valid in-schema call", "halluc_call": "parseable, hallucinated tool",
            "unparseable": "unparseable text", "prose": "prose / no call",
            "rejected": "non-response (serving)"}
    for cat in cats:
        plt.bar(labels, fracs[cat], bottom=bottom, label=nice[cat], color=colors[cat])
        bottom = [b + f for b, f in zip(bottom, fracs[cat])]
    plt.ylabel("fraction of assistant turns")
    plt.xlabel("Qwen-Coder (by scale)  |  other families  |  cloud")
    plt.title("Tool-call outcome composition under serving-uniform text-tools (8 seeds)")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = os.path.join(HERE, "protocol_gradient.png")
    plt.savefig(out, dpi=150)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
