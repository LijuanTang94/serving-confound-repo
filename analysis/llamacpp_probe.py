import json, urllib.request, urllib.error
fn={"name":"filesystem_list_directory","description":"list a directory",
    "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}
tools=[{"type":"function","function":fn}]
body=json.dumps({"messages":[{"role":"user","content":"Call filesystem_list_directory on ./project/regions"}],
                 "tools":tools,"temperature":0.7,"max_tokens":200}).encode()
req=urllib.request.Request("http://localhost:8081/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
try:
    r=json.load(urllib.request.urlopen(req,timeout=120))
    msg=r["choices"][0]["message"]
    print("STATUS: 200 OK")
    print("native tool_calls:", bool(msg.get("tool_calls")))
    if msg.get("tool_calls"): print("  ->", json.dumps(msg["tool_calls"])[:200])
    print("content:", repr((msg.get("content") or "")[:200]))
except urllib.error.HTTPError as e:
    print(f"STATUS: HTTP {e.code}:", e.read().decode()[:250])
except Exception as e:
    print("ERR", type(e).__name__, e)
