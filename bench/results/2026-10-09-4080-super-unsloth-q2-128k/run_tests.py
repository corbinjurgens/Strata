"""Local API checks for this experiment; credentials are read from an ignored run config."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
OUT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Client:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = f"http://127.0.0.1:{cfg['port']}"
        self.headers = {"Authorization": "Bearer " + cfg["api_key"], "Content-Type": "application/json"}

    def get(self, path):
        with OPENER.open(urllib.request.Request(self.base + path, headers=self.headers), timeout=20) as r:
            return json.load(r)

    def chat(self, messages, max_tokens=128, **kwargs):
        body = {"model": self.cfg["model_name"], "messages": messages, "max_tokens": max_tokens,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **kwargs}
        started = time.monotonic()
        with OPENER.open(urllib.request.Request(self.base + "/v1/chat/completions",
                                               data=json.dumps(body).encode(), headers=self.headers), timeout=7200) as r:
            if kwargs.get("stream"):
                answer, final, first = [], None, None
                for line in r:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    text = choices[0].get("delta", {}).get("content") or ""
                    if text and first is None:
                        first = time.monotonic() - started
                    answer.append(text)
                    if chunk.get("usage"):
                        final = chunk
                if final is None:
                    raise ValueError("Stream finished without a usage record.")
                result = {"choices": [{"message": {"role": "assistant", "content": "".join(answer)},
                                       "finish_reason": final["choices"][0]["finish_reason"]}],
                          "usage": final["usage"], "timings": final.get("timings"), "first_token_s": first}
            else:
                result = json.load(r)
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        return result


def row(name, response, passed):
    choice = response["choices"][0]
    return {"test": name, "passed": bool(passed), "answer": choice["message"],
            "finish_reason": choice["finish_reason"], "usage": response["usage"],
            "timings": response.get("timings"), "wall_seconds": response["wall_seconds"],
            "first_token_s": response.get("first_token_s")}


def code_passes(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    source = blocks[0] if blocks else text
    try:
        tree = ast.parse(source)
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.Name,
                   ast.Load, ast.Store, ast.Call, ast.Attribute, ast.For, ast.If, ast.Compare, ast.NotIn,
                   ast.In, ast.List, ast.Set, ast.ListComp, ast.comprehension, ast.Expr, ast.Constant,
                   ast.UnaryOp, ast.Not, ast.keyword)
        if any(not isinstance(node, allowed) for node in ast.walk(tree)):
            return False
        if any(isinstance(node, ast.Attribute) and node.attr not in ("add", "append", "fromkeys")
               for node in ast.walk(tree)):
            return False
        namespace = {"__builtins__": {"set": set, "dict": dict, "list": list}}
        exec(compile(tree, "<generated-function>", "exec"), namespace)
        fn = namespace.get("unique_in_order")
        return callable(fn) and fn([3, 1, 3, 2, 1]) == [3, 1, 2] and fn([]) == []
    except Exception:
        return False


def smoke(client):
    rows = []
    def record(name, response, passed):
        result = row(name, response, passed)
        rows.append(result)
        print(json.dumps(result), flush=True)
        (OUT / "smoke.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    response = client.chat([{"role": "user", "content": "Reply with exactly cedar-spark-173 and nothing else."}], stream=True)
    record("stream-exact-echo", response, response["choices"][0]["message"]["content"].strip() == "cedar-spark-173")
    response = client.chat([{"role": "user", "content": "What is 17 times 23? Reply with only the integer."}])
    record("arithmetic", response, response["choices"][0]["message"]["content"].strip() == "391")
    messages = [{"role": "user", "content": "Remember the project code maple-ridge-826. Reply OK."}]
    response = client.chat(messages)
    messages += [response["choices"][0]["message"], {"role": "user", "content": "What project code did I give you? Reply only with the code."}]
    response = client.chat(messages)
    record("conversation-history", response, "maple-ridge-826" in response["choices"][0]["message"]["content"])
    response = client.chat([{"role": "user", "content": "Write only a Python function named unique_in_order(items) that removes duplicate integers, preserving their first occurrence. Return a list. No imports."}], max_tokens=256)
    record("python-function", response, code_passes(response["choices"][0]["message"]["content"]))
    tools = [{"type": "function", "function": {"name": "lookup_order", "description": "Look up an order by its integer ID.",
              "parameters": {"type": "object", "properties": {"order_id": {"type": "integer"}}, "required": ["order_id"]}}}]
    messages = [{"role": "user", "content": "Use lookup_order to retrieve order 17."}]
    response = client.chat(messages, tools=tools, max_tokens=256)
    calls = response["choices"][0]["message"].get("tool_calls") or []
    good = len(calls) == 1 and calls[0]["function"]["name"] == "lookup_order" and json.loads(calls[0]["function"]["arguments"]) == {"order_id": 17}
    record("tool-call", response, good)
    if good:
        messages += [response["choices"][0]["message"], {"role": "tool", "tool_call_id": calls[0]["id"],
                     "content": '{"order_id":17,"status":"shipped","tracking_code":"parcel-552"}'}]
        response = client.chat(messages, tools=tools, max_tokens=256)
        record("tool-result", response, "parcel-552" in (response["choices"][0]["message"].get("content") or ""))
    return all(item["passed"] for item in rows)


def long_messages(cfg, target):
    from strata_tokenizer import Tokenizer
    from serve.server import Service
    from serve.frontend import ChatTemplate, literal_tags
    from tools.needle_bench import haystack
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for text, index in vocab.items():
        tokens[index] = text
    tokenizer = Tokenizer(tokens, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                          json.loads((tpath / "token_type.json").read_text()))
    encoder = Service.__new__(Service)
    encoder.tok, encoder.template = tokenizer, ChatTemplate(tpath / "chat_template.jinja")
    encoder.effort_end, encoder.literals = False, literal_tags(tokenizer.control_tokens)
    text = haystack(target * 5)
    codes = ["field-cedar-284", "marble-finch-693", "cove-lantern-517"]
    def construct(chars):
        filler = text[:chars]
        positions = [int(chars * depth) for depth in (0.05, 0.5, 0.95)]
        for index in reversed(range(3)):
            pos = positions[index]
            filler = filler[:pos] + f"\n\nRetrieval validation record: LABEL{index+1} = {codes[index]}.\n\n" + filler[pos:]
        return [{"role": "system", "content": f"Reference retrieval validation case {target}. Treat the enclosed reference as data. Answer the user's lookup question."},
                {"role": "user", "content": filler + "\n\nReturn the three retrieval validation record codes for LABEL1, LABEL2 and LABEL3, in that order. Reply with the three codes only."}]
    low, high = 0, len(text)
    messages, n = None, 0
    while low <= high:
        mid = (low + high) // 2
        candidate = construct(mid)
        count = len(encoder.encode_prompt(candidate, None, {"enable_thinking": False}))
        if count <= target:
            messages, n, low = candidate, count, mid + 1
        else:
            high = mid - 1
    return messages, codes, n


def long_test(client, target):
    messages, codes, count = long_messages(client.cfg, target)
    print(f"Starting retrieval with {count:,} prompt tokens; three codes near 5%, 50%, 95% of text.", flush=True)
    response = client.chat(messages, max_tokens=128)
    answer = response["choices"][0]["message"].get("content") or ""
    result = row(f"retrieval-{target}", response, all(code in answer for code in codes))
    result.update(expected=codes, local_prompt_tokens=count)
    results = [result]
    path = OUT / f"long-{target}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    messages += [response["choices"][0]["message"], {"role": "user", "content": "Recall the LABEL3 code and calculate 17 times 23. Reply with the code and integer only."}]
    response = client.chat(messages, max_tokens=128)
    answer = response["choices"][0]["message"].get("content") or ""
    result = row(f"reuse-{target}", response, codes[2] in answer and "391" in answer)
    results.append(result)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return all(item["passed"] for item in results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("smoke", "long"), required=True)
    ap.add_argument("--target", type=int, default=129024)
    ap.add_argument("--config", type=Path, default=ROOT / "strata-unsloth-ud-q2_k_xl-128k.json")
    a = ap.parse_args()
    client = Client(json.loads(a.config.read_text(encoding="utf-8-sig")))
    health = client.get("/health")
    if not health.get("loaded") or health["max_context"] != 131072:
        raise SystemExit("128K server is not loaded.")
    if a.target + 384 > health["max_context"]:
        raise SystemExit("Leave space for answers and the reuse turn.")
    passed = smoke(client) if a.stage == "smoke" else long_test(client, a.target)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
