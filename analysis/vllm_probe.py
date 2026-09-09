#!/usr/bin/env python3
"""
Cross-stack probe #3: how does vLLM handle the SAME OpenAI tools= request that the
default harness sends? Mirrors llamacpp_probe.py (identical request body) but launches
and tears down a `vllm serve` process per model, so it must run on a Linux GPU box.

Why this exists: the paper shows Ollama gates tools= per model by a template flag
(Phi-3/Gemma -> HTTP 400 "does not support tools"), while llama.cpp serves the same
weights. vLLM is a third, production-grade serving layer with a *different* policy:
tool parsing is enabled at server launch (--enable-auto-tool-choice --tool-call-parser),
not per-model. We probe the DEFAULT path (plain `vllm serve <model>`, no tool flags),
which is what a harness user gets out of the box, and record the outcome in the same
vocabulary as crossstack.csv: 200_native / 200_text / HTTP4xx_<reason> / error.

Whatever vLLM does (silently drops to text, or refuses for lack of a configured
parser), it reinforces the thesis: the serving layer, not the model, determines what
the measurement records.

Usage (on the GPU box):
    pip install "vllm>=0.6.0"
    # ungated models work with no token; Llama/Gemma need: export HF_TOKEN=hf_...
    python vllm_probe.py                 # default model set (ungated: qwen, phi3)
    python vllm_probe.py --all           # add gated llama3.2 + gemma (needs HF_TOKEN)
    python vllm_probe.py --models Qwen/Qwen2.5-Coder-0.5B-Instruct:qwen2.5-coder:0.5b
Outputs: vllm_crossstack.csv  (paste this back; it becomes column 3 of Table 2)
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.error, urllib.request

PORT = 8000
BASE = f"http://localhost:{PORT}"

# label (matches crossstack.csv rows) -> HF model id.  gated=True needs HF_TOKEN + license.
DEFAULT_MODELS = [
    # label,                  hf_id,                                    gated
    ("qwen2.5-coder:0.5b", "Qwen/Qwen2.5-Coder-0.5B-Instruct",          False),
    ("phi3",               "microsoft/Phi-3-mini-4k-instruct",          False),
]
GATED_MODELS = [
    ("llama3.2",           "meta-llama/Llama-3.2-3B-Instruct",          True),
    # gemma-3 may not be in your vLLM build yet; gemma-2-2b-it is the family proxy.
    ("gemma3:270m",        "google/gemma-2-2b-it",                      True),
]
# Repro notes if the gated repos block you or you are on an older GPU:
#   * Meta/Google gating can take hours to approve. Ungated community re-uploads of the
#     same weights work immediately: unsloth/Llama-3.2-3B-Instruct, unsloth/gemma-2-2b-it.
#   * gemma-2 refuses float16 ("numerical instability"); on a GPU without bf16 (e.g. T4,
#     compute 7.5) launch it with --dtype float32. Llama-3.2 is fine with --dtype half.
#   Confirmed 2026-06-30: all of Qwen-0.5B, Phi-3, Llama-3.2, Gemma-2 return the byte-
#   identical default HTTP 400 (server-config check, precedes model dispatch).

# EXACT same request body as llamacpp_probe.py / the harness default.
FN = {"name": "filesystem_list_directory", "description": "list a directory",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"]}}
PROBE_BODY = {
    "messages": [{"role": "user",
                  "content": "Call filesystem_list_directory on ./project/regions"}],
    "tools": [{"type": "function", "function": FN}],
    "temperature": 0.7, "max_tokens": 200,
}


def wait_health(proc, timeout=600):
    """Poll /health until the server is up or the process dies."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=3)
            return True
        except Exception:
            time.sleep(3)
    return False


def probe(hf_id):
    body = dict(PROBE_BODY, model=hf_id)
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=180))
        msg = r["choices"][0]["message"]
        native = bool(msg.get("tool_calls"))
        detail = (json.dumps(msg["tool_calls"])[:160] if native
                  else repr((msg.get("content") or "")[:160]))
        return ("200_native" if native else "200_text"), detail
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode()[:200].replace("\n", " ")
        # normalize: vLLM refuses tool_choice=auto without a configured parser -> 400
        tag = "rejected" if e.code == 400 else f"http{e.code}"
        return f"HTTP{e.code}_{tag}", body_txt
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"


def run_one(label, hf_id):
    print(f"\n=== {label}  ({hf_id}) ===", flush=True)
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", hf_id, "--port", str(PORT),
           "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
           "--disable-log-requests"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not wait_health(proc):
            return "error", "server failed to start (see vLLM logs; check HF gating/VRAM)"
        outcome, detail = probe(hf_id)
        print(f"  outcome: {outcome}")
        print(f"  detail : {detail}")
        return outcome, detail
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        time.sleep(5)  # let VRAM free before next model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include gated llama3.2 + gemma")
    ap.add_argument("--models", nargs="*", default=None,
                    help="custom 'hf_id:label' pairs, overrides defaults")
    args = ap.parse_args()

    if args.models:
        # each arg is "hf_id:label" (label may itself contain colons, e.g. qwen2.5-coder:0.5b)
        models = []
        for p in args.models:
            hf_id, label = p.split(":", 1)
            models.append((label, hf_id, False))
    else:
        models = DEFAULT_MODELS + (GATED_MODELS if args.all else [])

    if args.all and not os.environ.get("HF_TOKEN"):
        print("WARN: --all needs HF_TOKEN (Llama/Gemma are gated). Continuing anyway.")

    rows = []
    for label, hf_id, _gated in models:
        outcome, detail = run_one(label, hf_id)
        rows.append((label, outcome, detail))

    out = "vllm_crossstack.csv"
    with open(out, "w") as f:
        f.write("model,vllm_tools,detail\n")
        for label, outcome, detail in rows:
            f.write(f'{label},{outcome},"{detail.replace(chr(34), chr(39))}"\n')
    print(f"\nWrote {out}:")
    for label, outcome, detail in rows:
        print(f"  {label:24s} {outcome}")
    print("\nPaste vllm_crossstack.csv back into the chat to add Table 2 column 3.")


if __name__ == "__main__":
    main()
