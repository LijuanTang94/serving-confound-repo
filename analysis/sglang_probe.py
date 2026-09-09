#!/usr/bin/env python3
"""Cross-stack probe #4: SGLang. Same OpenAI tools= request and vocabulary as
vllm_probe.py / llamacpp_probe.py. Like vLLM, SGLang enables tool-call parsing via a
launch flag (--tool-call-parser); we probe the DEFAULT launch (no parser) to mirror
the out-of-the-box harness path. SGLang is CUDA-only, so this needs a GPU box.

Usage (GPU box, e.g. free Colab T4):
    pip install "sglang[all]"
    python sglang_probe.py            # default: Qwen-0.5B + Phi-3 (ungated)
Outputs: sglang_crossstack.csv  (paste back to add a 4th cross-stack column)

Notes:
  * Launched with --attention-backend triton + --disable-cuda-graph + --dtype half so
    it starts on older GPUs (T4, compute 7.5) without flashinfer.
  * Whatever SGLang does by default (refuse for a missing parser, or drop the call to
    text) is a 4th config-level policy -> reinforces "measure the stack, not the model."

Result (2026-07-02, on an sm_80 A800 80GB): SGLang's DEFAULT launch accepts the tools=
request (HTTP 200) but returns the call as text (a fenced JSON block) rather than a
native tool_call, unless launched with --tool-call-parser. This is a 4th distinct
default policy: it differs from vLLM's default (which refuses outright with HTTP 400),
so the two modern production stacks disagree on the default. Confirmed on Qwen-0.5B
(call as a JSON text block) and Phi-3 (prose); the policy is a server-config property,
model-independent. On sm_80 (A800), Phi-3's fused JIT RMSNorm kernel crashed under the
default flashinfer path, so it was run with --attention-backend triton (Qwen was fine
either way); this is a kernel-build detail, not part of the measurement.

Hardware note: needs sm_80+ (Ampere or newer: A100 / L4 / A10 / RTX 3090 / RTX 4090).
It does NOT run on a Tesla T4 (sm_75): the bundled flashinfer RMSNorm kernel raises
KeyError('sm_75') and SGLang hard-imports flashinfer.sampling, a catch-22 on T4. On an
sm_80 box, ensure libnvrtc.so.13 is on LD_LIBRARY_PATH (sgl_kernel JIT-compiles the
sm_80 ops via nvrtc); on a China-network box set HF_ENDPOINT=https://hf-mirror.com.
"""
import glob, json, os, signal, subprocess, sys, time, urllib.request, urllib.error

PORT = 30000
BASE = f"http://localhost:{PORT}"
MODELS = [
    ("qwen2.5-coder:0.5b", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    ("phi3",               "microsoft/Phi-3-mini-4k-instruct"),
]
FN = {"name": "filesystem_list_directory", "description": "list a directory",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                     "required": ["path"]}}
BODY = {"messages": [{"role": "user",
                      "content": "Call filesystem_list_directory on ./project/regions"}],
        "tools": [{"type": "function", "function": FN}],
        "temperature": 0.7, "max_tokens": 200}


def cuda_env():
    libdirs = {d for d in glob.glob(
        "/usr/local/lib/python3.*/dist-packages/nvidia/**/lib", recursive=True)
        if os.path.isdir(d)}
    for td in glob.glob("/usr/local/lib/python3.*/dist-packages/torch/lib"):
        libdirs.add(td)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(sorted(libdirs)) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def wait(proc, t=1200):
    end = time.time() + t
    n = 0
    while time.time() < end:
        if proc.poll() is not None:
            return False
        for ep in ("/health", "/get_model_info", "/v1/models"):
            try:
                urllib.request.urlopen(BASE + ep, timeout=3)
                return True
            except Exception:
                pass
        n += 1
        if n % 6 == 0:
            print(f"   ...loading ({n*5}s)", flush=True)
        time.sleep(5)
    return False


def probe(hf):
    body = dict(BODY, model=hf)
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
        return f"HTTP{e.code}", e.read().decode()[:200].replace("\n", " ")
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"


def run_one(label, hf, env):
    print(f"\n=== {label}  ({hf}) ===", flush=True)
    logf = open(f"/tmp/sglang_{label.replace(':', '_').replace('/', '_')}.log", "w")
    cmd = [sys.executable, "-m", "sglang.launch_server",
           "--model-path", hf, "--port", str(PORT), "--host", "0.0.0.0",
           "--dtype", "half", "--disable-cuda-graph",
           "--attention-backend", "triton", "--mem-fraction-static", "0.8"]
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    try:
        if not wait(proc):
            logf.flush()
            tail = open(logf.name).read()[-1500:]
            print("  server did not start; log tail:\n", tail, flush=True)
            return "error", "server did not start"
        out, det = probe(hf)
        print(f"  outcome: {out}\n  detail : {det}", flush=True)
        return out, det
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT); proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        logf.close(); time.sleep(5)


def main():
    env = cuda_env()
    rows = [(label, *run_one(label, hf, env)) for label, hf in MODELS]
    with open("sglang_crossstack.csv", "w") as f:
        f.write("model,sglang_default,detail\n")
        for label, out, det in rows:
            f.write(f'{label},{out},"{det.replace(chr(34), chr(39))}"\n')
    print("\n===== paste sglang_crossstack.csv back =====")
    print("model,sglang_default")
    for label, out, det in rows:
        print(f"  {label:22s} {out}")


if __name__ == "__main__":
    main()
