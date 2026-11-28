"""R1 cost arm: steady-state throughput with continuous refill (fixes P5 Deviation 11).

Refill logic is pure/testable (RefillScheduler, window_throughput). The live path keeps
`concurrency` requests in flight for `duration_s`, replacing each completion immediately;
throughput is completions/tokens in [60s, duration-60s]. Single-request latency on 50 prompts
is measured sequentially against the same OpenAI endpoint (generation via httpx, as in
harness.generate; this module does not start servers).
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time
from pathlib import Path

def window_throughput(completion_times, duration_s=600, exclude_s=60):
    """Completions (monotone seconds from start) inside the steady window."""
    lo, hi = exclude_s, duration_s - exclude_s
    inside = [t for t in completion_times if lo <= t <= hi]
    window = hi - lo
    return {"window_s": window, "window_lo_s": lo, "window_hi_s": hi,
            "n_completions_in_window": len(inside),
            "throughput_req_s": (len(inside) / window) if window > 0 else 0.0}

class RefillScheduler:
    """Keeps `concurrency` in flight: completing one immediately submits the next pending item."""
    def __init__(self, concurrency=32):
        self.concurrency = concurrency
        self._in_flight: list[str] = []
        self.pending_submitted = 0
    @property
    def in_flight(self):
        return len(self._in_flight)
    def submit_initial(self, items):
        for it in list(items)[: self.concurrency]:
            self._in_flight.append(it); self.pending_submitted += 1
    def complete_one(self, next_item=None):
        done = self._in_flight.pop(0) if self._in_flight else None
        if next_item is not None:
            self._in_flight.append(next_item); self.pending_submitted += 1
        return done

async def _steady_async(endpoint, model, prompts, concurrency, duration_s):
    import httpx
    sched = RefillScheduler(concurrency)
    completions, token_counts = [], []
    t0 = time.monotonic()
    queue = list(prompts)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900)) as client:
        async def one(prompt):
            r = await client.post(endpoint.rstrip("/") + "/chat/completions", json={
                "model": model, "messages": prompt, "temperature": 0, "seed": 0, "max_tokens": 4096})
            r.raise_for_status()
            return (r.json().get("usage") or {}).get("completion_tokens", 0)
        in_flight = {}
        def launch():
            if not queue:
                return
            p = queue.pop(0)
            task = asyncio.ensure_future(one(p))
            in_flight[task] = p
            sched.pending_submitted += 1
        for _ in range(min(concurrency, len(queue))):
            launch()
        # continuous refill: as tasks finish, launch replacements until duration elapses;
        # prompt list cycles so the window never drains (card: refilled continuously)
        cycle = list(prompts)
        while time.monotonic() - t0 < duration_s:
            if not in_flight:
                queue.extend(cycle); launch(); continue
            done, _ = await asyncio.wait(set(in_flight), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                in_flight.pop(task, None)
                try:
                    token_counts.append(task.result())
                except Exception:
                    pass
                completions.append(time.monotonic() - t0)
                if not queue:
                    queue.extend(cycle)
                launch()
        return completions, token_counts

def steady_state(endpoint, model, prompts, concurrency=32, duration_s=600):
    completions, tokens = asyncio.run(_steady_async(endpoint, model, prompts, concurrency, duration_s))
    r = window_throughput(completions, duration_s)
    r.update({"concurrency": concurrency, "duration_s": duration_s,
              "total_completions": len(completions), "mean_out_tokens": (sum(tokens) / len(tokens) if tokens else None)})
    return r

def single_request_latency(endpoint, model, prompts, n=50):
    """Sequential latency on first n prompts; prefill/decode split from usage + wall time is
    approximate (server does not expose per-phase timing) and labelled as such in output."""
    import httpx
    rows = []
    with httpx.Client(timeout=httpx.Timeout(900)) as client:
        for prompt in list(prompts)[:n]:
            t0 = time.monotonic()
            r = client.post(endpoint.rstrip("/") + "/chat/completions", json={
                "model": model, "messages": prompt, "temperature": 0, "seed": 0, "max_tokens": 4096})
            dt = time.monotonic() - t0
            u = (r.json().get("usage") or {}) if r.status_code == 200 else {}
            rows.append({"latency_s": dt, "prompt_tokens": u.get("prompt_tokens"),
                         "completion_tokens": u.get("completion_tokens")})
    ok = [r for r in rows if r["completion_tokens"]]
    return {"n": len(rows), "mean_latency_s": (sum(r["latency_s"] for r in ok) / len(ok) if ok else None),
            "decode_tok_s_approx": (sum(r["completion_tokens"] for r in ok) / sum(r["latency_s"] for r in ok) if ok else None),
            "prefill_note": "prefill vs decode split not exposed by server; latency_s is end-to-end, "
                            "decode_tok_s_approx = completion_tokens/latency (upper bound on decode time share)",
            "rows": rows}

def main(argv=None):
    ap = argparse.ArgumentParser(prog="r1_cost.py")
    ap.add_argument("--endpoint", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32); ap.add_argument("--duration-s", type=int, default=600)
    ap.add_argument("--latency-n", type=int, default=50)
    a = ap.parse_args(argv)
    items = [json.loads(l) for l in Path(a.items).read_text().splitlines() if l.strip()]
    prompts = [it["messages"] for it in items]
    res = {"steady": steady_state(a.endpoint, a.model, prompts, a.concurrency, a.duration_s),
           "latency": single_request_latency(a.endpoint, a.model, prompts, a.latency_n)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["steady"], indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
