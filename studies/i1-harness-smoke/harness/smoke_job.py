#!/usr/bin/env python3
"""I1 harness smoke job: send 40 frozen arithmetic prompts to the unit's own vLLM.

Resumable at item granularity: items already in receipts/results.jsonl are skipped,
so a revoked-and-resumed unit must end with each id exactly once.
Pacing of 1.5 s per item gives the claim-revocation test a window; it is not a timing measurement.
"""
import json, os, sys, time, urllib.request

here = os.path.dirname(os.path.abspath(__file__))
base = os.environ["IDLEWORKS_BASE_URL"].rstrip("/")
model = os.environ["IDLEWORKS_MODEL"]
attempt = os.environ.get("IDLEWORKS_ATTEMPT", "1")
out_path = os.path.join(os.environ["RL_RECEIPT_DIR"], "receipts", "results.jsonl")
prompts = [json.loads(l) for l in open(os.path.join(here, "prompts.jsonl")) if l.strip()]
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["id"] for l in open(out_path) if l.strip()}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with open(out_path, "a") as out:
    for p in prompts:
        if p["id"] in done:
            continue
        body = {"model": model, "messages": [{"role": "user", "content": p["prompt"]}],
                "temperature": 0, "max_tokens": 32, "seed": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        resp = json.load(opener.open(req, timeout=120))
        dt = time.time() - t0
        text = resp["choices"][0]["message"]["content"].strip()
        usage = resp.get("usage") or {}
        out.write(json.dumps({"id": p["id"], "answer": text, "expected": p["answer"],
                              "correct": text == p["answer"], "attempt": attempt,
                              "prompt_tokens": usage.get("prompt_tokens"),
                              "completion_tokens": usage.get("completion_tokens"),
                              "latency_s": round(dt, 4)}) + "\n")
        out.flush()
        print(f"{p['id']} {'ok' if text == p['answer'] else 'miss'} {dt:.2f}s", flush=True)
        time.sleep(1.5)
print(f"done: {len(prompts)} items, attempt {attempt}", flush=True)
