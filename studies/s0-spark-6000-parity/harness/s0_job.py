#!/usr/bin/env python3
"""S0 unit job: greedy generations for the frozen prompt set against the unit's own vLLM server.

Stdlib only. Run by idleworks_unit.py after the server is healthy. Reads
IDLEWORKS_BASE_URL, IDLEWORKS_MODEL, IDLEWORKS_GPU_UUID, IDLEWORKS_CONTAINER,
RL_RECEIPT_DIR and IDLEWORKS_ATTEMPT from the environment.

Writes, under $RL_RECEIPT_DIR/receipts/:
  results.jsonl        one JSON line per finished item (append only; items already present
                       are skipped, so an interrupted unit resumes where it stopped)
  treatment-a<N>.json  what this attempt actually ran on: unit label, concurrency, CPU
                       architecture, GPU name/UUID/driver/compute capability, and the serving
                       container's image ID, VLLM_* environment and arguments (docker inspect),
                       plus the server's /v1/models reply

The request body is the one harness.generate (RonanLabsAI/mom-lab) sends for a greedy,
non-thinking run with a 4,096-token override: model, messages, temperature 0.0,
top_p 1.0, seed 0, max_tokens 4096, chat_template_kwargs {"enable_thinking": false}.
Benchmarks run one after another in file order (gsm8k, math500, ifeval), each with at
most --concurrency requests in flight, as harness.generate does.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
BENCH_ORDER = ["gsm8k", "math500", "ifeval"]
PROMPTS_SHA256 = "1d0ef52f6e1f6a158c6e6df87da13a8b2a298c1f6df450efc5daa8ba97997933"
N_PROMPTS = 1179


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def call(url, body, retries, timeout):
    last = None
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            return post(url, body, timeout), time.monotonic() - t0
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code not in RETRY_STATUS:
                break
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(min(60.0, 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(last)


def payload(item, model):
    return {
        "model": model,
        "messages": item["messages"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": int(item["max_tokens"]),
        "chat_template_kwargs": {"enable_thinking": False},
    }


def run_json(argv):
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return out.stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {e}"


def treatment(args, base, attempt):
    uuid = os.environ.get("IDLEWORKS_GPU_UUID", "")
    container = os.environ.get("IDLEWORKS_CONTAINER", "")
    t = {"unit": args.unit, "concurrency": args.concurrency, "attempt": attempt,
         "machine": platform.machine(), "gpu_uuid": uuid, "container": container}
    smi = run_json(["nvidia-smi", "-i", uuid, "--query-gpu=name,uuid,driver_version,compute_cap",
                    "--format=csv,noheader"]).strip()
    parts = [p.strip() for p in smi.split(",")]
    t["gpu"] = dict(zip(["name", "uuid", "driver", "compute_cap"], parts)) if len(parts) == 4 else {"raw": smi}
    raw = run_json(["docker", "inspect", container])
    try:
        ins = json.loads(raw)[0]
        t["image_id"] = ins.get("Image")
        t["vllm_env"] = sorted(e for e in ins["Config"].get("Env") or [] if e.startswith("VLLM_"))
        t["args"] = ins.get("Args") or []
    except (ValueError, KeyError, IndexError, TypeError):
        t["docker_inspect_error"] = raw[:500]
    try:
        with urllib.request.urlopen(base + "/models", timeout=30) as r:
            t["server_models"] = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        t["server_models_error"] = str(e)
    return t


def read_done(out_path):
    """Ids already recorded. A line that does not parse (for example a final line cut short by
    an interruption) is skipped and counted; the item is then requested again."""
    done, bad = set(), 0
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (ValueError, KeyError, TypeError):
                bad += 1
    return done, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=2700.0)
    args = ap.parse_args(argv)

    raw = Path(args.prompts).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROMPTS_SHA256:
        sys.exit("prompts.jsonl does not match the frozen SHA-256")
    items = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    if len(items) != N_PROMPTS:
        sys.exit(f"expected {N_PROMPTS} prompts, found {len(items)}")

    base = os.environ["IDLEWORKS_BASE_URL"].rstrip("/")
    model = os.environ["IDLEWORKS_MODEL"]
    attempt = int(os.environ.get("IDLEWORKS_ATTEMPT") or 1)
    rdir = Path(os.environ["RL_RECEIPT_DIR"]) / "receipts"
    rdir.mkdir(parents=True, exist_ok=True)
    out_path = rdir / "results.jsonl"

    (rdir / f"treatment-a{attempt}.json").write_text(json.dumps(treatment(args, base, attempt), indent=1))
    done, bad = read_done(out_path)
    if bad:
        print(f"warning: {bad} unparseable line(s) in results.jsonl skipped", file=sys.stderr, flush=True)
    # Start on a fresh line if an earlier attempt was cut off mid-line.
    if out_path.exists() and out_path.stat().st_size and not out_path.read_bytes().endswith(b"\n"):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n")

    url = base + "/chat/completions"
    lock = threading.Lock()
    fout = open(out_path, "a", encoding="utf-8")
    n_fail = 0

    def one(item):
        nonlocal n_fail
        try:
            data, lat = call(url, payload(item, model), args.retries, args.timeout)
            ch = data["choices"][0]
            usage = data.get("usage") or {}
            rec = {
                "id": item["id"],
                "bench": item["bench"],
                "output": (ch.get("message") or {}).get("content"),  # null if absent, never ""
                "finish_reason": ch.get("finish_reason"),
                "n_out_tokens": usage.get("completion_tokens"),
                "n_prompt_tokens": usage.get("prompt_tokens"),
                "latency_s": round(lat, 4),
                "attempt": attempt,
            }
            # Same schema analyze.py enforces. A reply that fails it is not recorded, so the item
            # counts as a failure and is requested again on resume.
            if not (isinstance(rec["output"], str) and rec["finish_reason"] in ("stop", "length")
                    and isinstance(rec["n_out_tokens"], int) and isinstance(rec["n_prompt_tokens"], int)):
                raise RuntimeError(f"reply failed schema: {json.dumps(rec)[:300]}")
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
        except Exception as e:  # not written, so it is retried on resume
            with lock:
                n_fail += 1
            print(f"FAIL {item['id']}: {e}", file=sys.stderr, flush=True)

    t_start = time.time()
    try:
        for bench in BENCH_ORDER:
            todo = [it for it in items if it["bench"] == bench and it["id"] not in done]
            print(f"[{bench}] {sum(1 for it in items if it['bench'] == bench)} items, {len(todo)} to run",
                  flush=True)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(one, todo))
    finally:
        fout.close()
        (rdir / f"timing-a{attempt}.json").write_text(json.dumps(
            {"attempt": attempt, "start_unix": round(t_start, 3), "end_unix": round(time.time(), 3),
             "failures": n_fail}))

    print(f"done, failures this attempt: {n_fail}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
