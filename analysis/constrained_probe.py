import json, urllib.request, urllib.error, sys
sys.path.insert(0, "inference")
import importlib.util
spec = importlib.util.spec_from_file_location("rr", "inference/run_react.py")
# avoid importing whole module (needs fire); just inline the parser
import re, uuid
def parse(content):
    cands=[]
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL): cands.append(m.group(1))
    if not cands:
        m=re.search(r"(\{.*\"name\".*\})", content, re.DOTALL)
        if m: cands.append(m.group(1))
    out=[]
    for raw in cands:
        try: o=json.loads(raw)
        except: continue
        if isinstance(o,dict) and "name" in o: out.append(o["name"])
    return out

NAMES=['claim_done_claim_done','filesystem_create_directory','filesystem_directory_tree','filesystem_edit_file','filesystem_get_file_info','filesystem_list_allowed_directories','filesystem_list_directory','filesystem_list_directory_with_sizes','filesystem_move_file','filesystem_read_file','filesystem_read_media_file','filesystem_read_multiple_files','filesystem_read_text_file','filesystem_search_files','filesystem_write_file','python_execute_python_execute']
SCHEMA={"type":"object","properties":{"name":{"type":"string","enum":NAMES},"arguments":{"type":"object"}},"required":["name","arguments"]}
RF={"type":"json_schema","json_schema":{"name":"tool_call","strict":True,"schema":SCHEMA}}
TASK=("You are working in a Python project under ./project . project/regions/ has region modules each "
      "defining TAX_RATE, ACTIVE, TIER. Compute aggregates and write ./answers.json, then call claim_done. "
      "Respond with a single tool call.")
N=8
print(f"{'model':22s} valid-in-schema / trials   (constrained decoding)")
for m in ["qwen2.5-coder:0.5b","qwen2.5-coder:1.5b","qwen2.5-coder:3b","qwen2.5-coder:7b","qwen2.5-coder:14b","llama3.2:latest","phi3:latest","gemma3:4b","gemma3:270m"]:
    ok=0; tot=0
    for _ in range(N):
        body=json.dumps({"model":m,"messages":[{"role":"user","content":TASK}],"response_format":RF,"temperature":1.0,"max_tokens":300}).encode()
        req=urllib.request.Request("http://localhost:11434/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
        try:
            r=json.load(urllib.request.urlopen(req,timeout=120))
            c=r["choices"][0]["message"].get("content","") or ""
            tot+=1
            nm=parse(c)
            if nm and nm[0] in NAMES: ok+=1
        except Exception as e:
            tot+=1
    print(f"{m:22s} {ok:2d}/{tot:<2d}")
