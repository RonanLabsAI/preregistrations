"""Vendor headline check: Ollama's published DGX Spark gpt-oss:120b prefill/decode rates.

Measures, through a LiteLLM route name, sequential (concurrency 1) streamed
chat completions of the vendor's own workload (first 200 lines of the
public-domain text the vendor used, 500 output tokens, temperature 0, seed 0).
Each long trial is paired with a short control trial so that fixed router and
network overhead cancels in the prefill estimate. A unique per-trial prefix
defeats prompt-prefix caching, which the vendor disabled by unloading the
model between runs (unloading a production resident is not permitted here).

Standard library only. The router key is read from a local file and never
logged. See PREREG.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import k3  # noqa: E402

PRODUCT = "vendor-headline-check-v1"
TITLE = "Vendor headline check: Ollama's DGX Spark gpt-oss:120b prefill and decode rates"
LOCK = HERE / "inputs.lock.json"
CODE_FILES = [HERE / "bench.py", HERE.parent / "common" / "k3.py", LOCK]
BAND = 0.10  # preregistered +/-10% equivalence band around each vendor number
SHORT_PROMPT = "hi"
SHORT_MAX_TOKENS = 16
MIN_LONG_COMPLETION = 100


def code_sha() -> str:
    return hashlib.sha256(b"".join(p.read_bytes() for p in CODE_FILES)).hexdigest()[:12]


def median_ci(values: list[float]) -> tuple[float, list[float], float]:
    """Median with a distribution-free order-statistic CI (coverage >= 95%)."""
    xs = sorted(values)
    n = len(xs)
    if n < 6:
        return (statistics.median(xs) if xs else math.nan), [math.nan, math.nan], 0.0
    # smallest symmetric k with binomial coverage >= 0.95
    best = None
    for k in range(n // 2, 0, -1):
        cov = sum(math.comb(n, i) for i in range(k, n - k + 1)) / 2 ** n
        if cov >= 0.95:
            best = (k, cov)
            break
    k, cov = best if best else (1, 1.0)
    return statistics.median(xs), [xs[k - 1], xs[n - k]], cov


def classify(ci: list[float], claim: float) -> str:
    lo, hi = claim * (1 - BAND), claim * (1 + BAND)
    if any(math.isnan(x) for x in ci):
        return "inconclusive"
    if ci[1] < lo:
        return "below claim"
    if ci[0] > hi:
        return "above claim"
    if lo <= ci[0] and ci[1] <= hi:
        return "consistent with claim"
    return "inconclusive"


def stream_chat(base: str, key: str, route: str, content: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps({
        "model": route, "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0, "seed": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = None
    deltas = 0
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.perf_counter()
                    t_first = t_first if t_first is not None else now
                    t_last = now
                    deltas += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.perf_counter()
    return {"ttft_s": (t_first - t0) if t_first is not None else None,
            "stream_s": (t_last - t_first) if t_first is not None else None,
            "total_s": t_end - t0, "deltas": deltas, "usage": usage, "finish_reason": finish}


def guard_inflight(cmd: list[str] | None) -> int | None:
    """Foreign in-flight requests on the serving engine, from the frozen read-only probe."""
    if not cmd:
        return None
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout
    status = json.loads(out)
    return sum(int(s.get("inflight") or 0) for s in (status.get("services") or {}).values())


def run_trials(args, key: str, pkg: k3.Package | None, prompt: str) -> list[dict]:
    guard = json.loads(args.guard_cmd) if args.guard_cmd else None
    rows = []
    valid_long = 0
    attempts = 0
    while valid_long < args.runs and attempts < args.runs + args.max_extra:
        attempts += 1
        for kind in ("short", "long"):
            nonce = secrets.token_hex(4)
            content = f"[{nonce}] " + (SHORT_PROMPT if kind == "short" else prompt)
            before = guard_inflight(guard)
            res = stream_chat(args.base, key, args.route, content, SHORT_MAX_TOKENS if kind == "short" else args.max_tokens, args.timeout)
            after = guard_inflight(guard)
            usage = res["usage"] or {}
            ok = (res["ttft_s"] is not None and usage.get("prompt_tokens") and usage.get("completion_tokens")
                  and not (before or 0))
            if kind == "long":
                ok = ok and usage.get("completion_tokens", 0) >= MIN_LONG_COMPLETION
            row = {"trial": attempts, "kind": kind, "nonce": nonce, "ok": bool(ok),
                   "foreign_inflight_before": before, "foreign_inflight_after": after, **res}
            if kind == "long" and ok and res["stream_s"]:
                row["decode_tok_s"] = (usage["completion_tokens"] - 1) / res["stream_s"]
            rows.append(row)
            if pkg:
                pkg.log("trial", trial=attempts, kind=kind, ok=bool(ok), ttft_s=res["ttft_s"],
                        completion_tokens=usage.get("completion_tokens"), prompt_tokens=usage.get("prompt_tokens"))
            if kind == "short" and not ok:
                break
        if rows[-1]["kind"] == "long" and rows[-1]["ok"] and rows[-2]["ok"]:
            valid_long += 1
    return rows


def summarize(rows: list[dict], claim: dict) -> dict:
    pairs = []
    for t in sorted({r["trial"] for r in rows}):
        s = [r for r in rows if r["trial"] == t and r["kind"] == "short"]
        lg = [r for r in rows if r["trial"] == t and r["kind"] == "long"]
        if s and lg and s[0]["ok"] and lg[0]["ok"]:
            pairs.append((s[0], lg[0]))
    decode = [lg["decode_tok_s"] for _, lg in pairs]
    prefill = []
    for s, lg in pairs:
        dp = lg["usage"]["prompt_tokens"] - s["usage"]["prompt_tokens"]
        dt = lg["ttft_s"] - s["ttft_s"]
        if dt > 0:
            prefill.append(dp / dt)
    d_med, d_ci, d_cov = median_ci(decode)
    p_med, p_ci, p_cov = median_ci(prefill)
    return {
        "n_valid_pairs": len(pairs), "n_trials": len({r["trial"] for r in rows}),
        "decode": {"median": d_med, "ci": d_ci, "coverage": d_cov, "n": len(decode),
                   "claim": claim["decode_tok_s"], "classification": classify(d_ci, claim["decode_tok_s"])},
        "prefill": {"median": p_med, "ci": p_ci, "coverage": p_cov, "n": len(prefill),
                    "claim": claim["prefill_tok_s"], "classification": classify(p_ci, claim["prefill_tok_s"])},
        "long_prompt_tokens": sorted({lg["usage"]["prompt_tokens"] for _, lg in pairs}),
        "median_ttft_long_s": statistics.median([lg["ttft_s"] for _, lg in pairs]) if pairs else None,
        "median_ttft_short_s": statistics.median([s["ttft_s"] for s, _ in pairs]) if pairs else None,
    }


def verdict(summary: dict, prereg_url: str, fp: str) -> str:
    d, p = summary["decode"], summary["prefill"]
    return "\n".join([
        "# Verdict: Ollama's DGX Spark gpt-oss headline on our stack",
        "",
        f"Preregistration: {prereg_url}",
        "",
        f"Decode: **{d['classification']}** under the preregistered equivalence band.",
        (f"Median client-side decode rate was {d['median']:.2f} tokens/s "
         f"(order-statistic {100 * d['coverage']:.1f}% CI [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}], n={d['n']} runs) "
         f"against the vendor's {d['claim']:.2f} tokens/s, from receipts/results.jsonl."),
        f"Prefill: **{p['classification']}** under the preregistered equivalence band.",
        (f"Median overhead-differenced prefill rate was {p['median']:.0f} tokens/s "
         f"(order-statistic {100 * p['coverage']:.1f}% CI [{p['ci'][0]:.0f}, {p['ci'][1]:.0f}], n={p['n']} paired runs) "
         f"against the vendor's {p['claim']:.0f} tokens/s, from receipts/results.jsonl."),
        (f"Valid paired trials were {summary['n_valid_pairs']}/{summary['n_trials']} attempted trials, "
         "from receipts/results.jsonl."),
        "",
        "## What this does and does not establish",
        "",
        "It tests the vendor's single-stream numbers for the same model weights and workload on the same hardware class, "
        "as served by our current production engine version and configuration, measured at the client through the router. "
        "Client-side decode timing includes streaming transport; it can only understate the engine's own rate. "
        "It does not test the vendor's older engine version, driver or an unloaded-model setup, and it does not measure "
        "multi-user throughput. The environment fingerprint is recorded in manifest.json.",
        "",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="router base URL (route names only)")
    ap.add_argument("--route", default="gpt-oss")
    ap.add_argument("--key-file", type=Path)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--max-extra", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--guard-cmd", help="JSON argv of the frozen read-only executor status probe")
    ap.add_argument("--engine-fingerprint", help="frozen serving-engine identity (from the probe)")
    ap.add_argument("--expect-fingerprint")
    ap.add_argument("--print-fingerprint", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--prereg", type=Path, default=HERE / "PREREG.md")
    ap.add_argument("--prereg-url")
    ap.add_argument("--arm", default="vendor")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    fp = f"{args.engine_fingerprint or 'engine-unset'}|route={args.route}|c1|code@{code_sha()}"
    if args.print_fingerprint:
        print(fp)
        return 0
    k3.check_fingerprint(fp, args.expect_fingerprint)
    lock = json.loads(LOCK.read_text())
    prompt_path = HERE / "prompt.txt"
    if k3.sha256_file(prompt_path) != lock["frozen_files"]["prompt.txt"]["sha256"]:
        raise SystemExit("KILL: prompt drift")
    ident = k3.identity_from_env(PRODUCT)
    out = args.out or (Path(ident["receipt_dir"]) if ident["receipt_dir"] else None)
    if not (out and args.base and args.key_file and args.prereg_url):
        ap.error("--base, --key-file, --prereg-url and --out (or RL_RECEIPT_DIR) are required")
    key = args.key_file.read_text().strip()
    pkg = k3.Package(out)
    pkg.log("start", fingerprint=fp, runs=args.runs, route=args.route, unit=ident["unit_id"], pid=os.getpid())
    pkg.add_frozen_input("receipts/inputs/prompt.txt", prompt_path, lock["frozen_files"]["prompt.txt"]["sha256"])
    pkg.add_receipt("receipts/inputs.lock.json", LOCK.read_bytes())
    rows = run_trials(args, key, pkg, prompt_path.read_text())
    summary = summarize(rows, lock["claim"])
    pkg.write_jsonl("receipts/results.jsonl", [{"kind": "summary", **summary}] + [{"kind": "trial", **r} for r in rows])
    pkg.log("done", decode=summary["decode"]["classification"], prefill=summary["prefill"]["classification"])
    env = {"hardware": "NVIDIA DGX Spark (GB10)", "engine": args.engine_fingerprint or "unknown", "fingerprint": fp,
           "client": k3.host_summary()}
    pkg.finish(identity=ident, title=TITLE, lane="L3", prereg_path=args.prereg, prereg_url=args.prereg_url,
               arm=args.arm, seed=args.seed, environment=env, verdict=verdict(summary, args.prereg_url, fp),
               tiers=["P1", "P3"])
    print(json.dumps({"ok": True, "decode": summary["decode"]["classification"],
                      "prefill": summary["prefill"]["classification"], "valid": summary["n_valid_pairs"]}))
    return 0 if summary["n_valid_pairs"] >= args.runs else 3


if __name__ == "__main__":
    sys.exit(main())
