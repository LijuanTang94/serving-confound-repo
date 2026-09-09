#!/usr/bin/env python3
"""Third task: does the serving-layer confound replicate on HumanEval (a widely-used
benchmark)? Single-turn per problem (the model is asked to emit ONE tool call that
submits its solution), so there is no agentic loop and no non-termination on weak
local models.

Two measurements, matching the paper:
  (1) native gating: send OpenAI tools= per model, record accept-as-text / native /
      HTTP 400 rejection (expect the same per-model policy as Tables 1-3).
  (2) text-tools fidelity: drop tools=, put the tool list + strict JSON call protocol
      + allowed-name list in the prompt, parse the call from text, score a valid
      in-schema call (parseable AND name == submit_solution). Per-problem rate.

Needs Ollama up (LOCA_BASE default http://localhost:11434/v1) and humaneval6.json
(6 HumanEval problems: [{task_id, prompt, entry_point, ...}]).
Writes humaneval_probe.csv.
"""
import gzip, json, os, re, urllib.request, urllib.error

BASE = os.environ.get("LOCA_BASE", "http://localhost:11434/v1")
N_PROBLEMS = 6


def load_problems():
    """Load the first N HumanEval problems; download them (MIT, openai/human-eval)
    if not cached locally, so we do not redistribute the benchmark data."""
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humaneval6.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    raw = gzip.decompress(urllib.request.urlopen(url, timeout=60).read()).decode()
    probs = [json.loads(l) for l in raw.splitlines()][:N_PROBLEMS]
    json.dump(probs, open(cache, "w"))
    return probs


PROBLEMS = load_problems()

TOOL = {"name": "submit_solution",
        "description": "Submit the completed Python function implementation.",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string",
                                               "description": "the full function source"}},
                       "required": ["code"]}}
TOOLS = [{"type": "function", "function": TOOL}]
ALLOWED = {"submit_solution"}

GATING_MODELS = ["qwen2.5-coder:0.5b", "llama3.2:latest", "phi3:latest", "gemma3:270m"]
FIDELITY_MODELS = ["qwen2.5-coder:0.5b", "qwen2.5-coder:1.5b", "qwen2.5-coder:3b",
                   "llama3.2:latest", "phi3:latest", "gemma3:4b", "gemma3:270m"]


def post(model, messages, tools=None, max_tokens=400):
    body = {"model": model, "messages": messages, "temperature": 1.0,
            "top_p": 1.0, "max_tokens": max_tokens}
    if tools is not None:
        body["tools"] = tools
    req = urllib.request.Request(f"{BASE}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=240))
        return 200, d["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:150].replace("\n", " ")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def task_prompt(p):
    return ("Complete the following Python function, then submit your implementation "
            "by calling the submit_solution tool with the full source as `code`.\n\n"
            f"```python\n{p['prompt']}```")


TEXT_PROTOCOL = (
    "\n\nYou have one tool:\n"
    "  submit_solution(code: string) - submit the completed Python function.\n"
    "To call it, output EXACTLY one JSON object on its own line:\n"
    '  {"name": "submit_solution", "arguments": {"code": "<full source>"}}\n'
    "The name MUST be exactly one of: submit_solution.")


def parse_text_call(text):
    """Return 'valid' | 'hallucinated' | 'unparseable' for a text tool call."""
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}', text, re.S):
        name = m.group(1)
        return "valid" if name in ALLOWED else "hallucinated"
    # arguments may nest braces; fall back to a name-key search
    m = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if m:
        return "valid" if m.group(1) in ALLOWED else "hallucinated"
    return "unparseable"


def main():
    print("=== (1) native gating on HumanEval ===")
    gating = {}
    for model in GATING_MODELS:
        code, msg = post(model, [{"role": "user", "content": task_prompt(PROBLEMS[0])}],
                         tools=TOOLS, max_tokens=50)
        if code == 200:
            out = "200_native" if msg.get("tool_calls") else "200_text"
        elif code == 400:
            out = "HTTP400_rejected"
        else:
            out = f"err{code}"
        gating[model] = out
        print(f"  {model:22s} {out}")

    print("\n=== (2) text-tools per-problem fidelity on HumanEval ===")
    rows = []
    for model in FIDELITY_MODELS:
        outcomes = []
        for p in PROBLEMS:
            content = task_prompt(p) + TEXT_PROTOCOL
            code, msg = post(model, [{"role": "user", "content": content}],
                             tools=None, max_tokens=400)
            if code != 200 or not isinstance(msg, dict):
                outcomes.append("error"); continue
            outcomes.append(parse_text_call(msg.get("content") or ""))
        n = len(outcomes)
        valid = sum(o == "valid" for o in outcomes)
        rate = valid / n if n else 0.0
        rows.append((model, valid, n, rate, gating.get(model, "-")))
        print(f"  {model:22s} valid {valid}/{n} = {rate*100:3.0f}%   "
              f"({','.join(outcomes)})")

    with open("humaneval_probe.csv", "w") as f:
        f.write("model,native_gating,text_tools_valid,n,text_tools_rate\n")
        for model in GATING_MODELS:
            if model not in FIDELITY_MODELS:
                f.write(f"{model},{gating[model]},,,\n")
        for model, valid, n, rate, gate in rows:
            f.write(f"{model},{gate},{valid},{n},{rate:.3f}\n")
    print("\nwrote humaneval_probe.csv")


if __name__ == "__main__":
    main()
