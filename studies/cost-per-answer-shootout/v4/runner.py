"""Frozen runner for cost-per-answer-shootout v2 (protocol v4). Private details come from a runtime file.

Same models, prompt, engine flags, inputs, costing and intervals as the v1 study (protocol v3). Two changes:
1. Primary scoring uses `extract_final_number`, a robust final-answer extractor frozen before any v2 output.
   The v1 strict scorer is retained as a secondary outcome on the same responses.
2. Requests go straight to the engine this runner launched on its claimed GPU (loopback), not through the
   research router. No router credential is used or needed.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
STOP = threading.Event()
PRODUCT = "cost-per-answer-shootout-v2"
SYSTEM = "Answer the math problem. Return only the final number. /no_think"
CONTAINER = "w18-cost-v2-engine"
SERVED = "w18-cost-v2"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def wilson(k, n):
    if not n:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [0.0 if k == 0 else max(0, c - r), 1.0 if k == n else min(1, c + r)]


def normalized_number(text):
    """v1 strict scorer, unchanged: the whole reply must be one number (commas and $ allowed)."""
    text = text.split("####")[-1].strip().replace(",", "").replace("$", "")
    m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    return float(text) if m else None


# v2 primary extractor. Frozen in protocol v4 before any v2 output exists.
_NUM = r"(?:-\$?|\$-?)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUM_RE = re.compile(r"(?<![\w.])" + _NUM)
_HASH = re.compile(r"####\s*(" + _NUM + ")")
_BOXED = re.compile(r"\\boxed\{\s*(" + _NUM + r")")
_ANSWER = re.compile(r"\b(?:final answer|answer)\b\s*(?:is|=|:)?", re.IGNORECASE)


def _to_float(token):
    value = float(token.replace("$", "").replace(",", "").replace(" ", ""))
    return value if math.isfinite(value) else None


def extract_final_number(text, finish_reason="stop"):
    """Return (value, rule). value is None when no answer qualifies.

    Order: last '#### N'; last '\\boxed{N}'; after the last 'answer'/'final answer' marker, the last number
    on that line (or, if that line has none, the first number after it); a reply that is only a number;
    otherwise the last number in the reply. A reply cut off by the output cap (finish_reason 'length')
    qualifies only through an explicit marker, never through the bare or last-number fallbacks.
    """
    t = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)
    for token in ["**", "__", "\\(", "\\)", "\\[", "\\]"]:
        t = t.replace(token, " ")
    found = _HASH.findall(t)
    if found:
        return _to_float(found[-1]), "hash-delimiter"
    found = _BOXED.findall(t)
    if found:
        return _to_float(found[-1]), "boxed"
    markers = list(_ANSWER.finditer(t))
    if markers:
        rest = t[markers[-1].end() :]
        line = rest.split("\n", 1)[0]
        numbers = _NUM_RE.findall(line)
        if numbers:
            return _to_float(numbers[-1]), "answer-marker"
        numbers = _NUM_RE.findall(rest)
        if numbers:
            return _to_float(numbers[0]), "answer-marker-next"
    if finish_reason == "length":
        return None, "truncated-no-marker"
    strict = normalized_number(t.strip().rstrip("."))
    if strict is not None:
        return strict, "bare-number"
    numbers = _NUM_RE.findall(t)
    if numbers:
        return _to_float(numbers[-1]), "last-number"
    return None, "no-number"


def same_number(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6


def ratio_ci(rows_by_id, num_key, den_key, seed=42):
    """Percentile bootstrap of cost-per-correct over question ids. Zero correct -> inf."""
    ids = sorted(rows_by_id)
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        pick = rng.choices(ids, k=len(ids))
        cost = sum(rows_by_id[i][num_key] for i in pick)
        correct = sum(rows_by_id[i][den_key] for i in pick)
        draws.append(cost / correct if correct else math.inf)
    draws.sort()
    return [draws[49], draws[1949]]


def paired_ratio_ci(a, b, seed=42):
    """a, b: {question_id: row}. Bootstrap of (cost/correct in a) / (cost/correct in b), same ids."""
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        pick = rng.choices(ids, k=len(ids))
        ka = sum(a[i]["correct"] for i in pick)
        kb = sum(b[i]["correct"] for i in pick)
        if not ka or not kb:
            draws.append(math.nan)
            continue
        ca = sum(a[i]["cost_usd"] for i in pick) / ka
        cb = sum(b[i]["cost_usd"] for i in pick) / kb
        draws.append(ca / cb)
    undefined = sum(math.isnan(x) for x in draws)
    finite = sorted(x for x in draws if not math.isnan(x))
    if undefined > 100 or not finite:
        return None, undefined
    # Undefined draws (a zero denominator) are reported, not imputed; >5% undefined -> no interval.
    lo = finite[int(0.025 * len(finite))]
    hi = finite[min(len(finite) - 1, int(0.975 * len(finite)))]
    return [lo, hi], undefined


class Run:
    def __init__(self, arm, smoke=False):
        self.product, self.arm, self.smoke = PRODUCT, arm, smoke
        self.runtime = json.loads(Path(os.environ["W18_RUNTIME"]).read_text())
        self.plan = json.loads((ROOT / "freeze.json").read_text())
        self.meta = json.loads((ROOT / "publication.json").read_text())[PRODUCT]
        self.runid = os.environ.get("RL_RUN_ID", f"smoke-{int(time.time())}")
        self.out = Path(os.environ["RL_RECEIPT_DIR"])
        self.out.mkdir(parents=True, exist_ok=True)
        for name in ["receipts", "figures", "drafts"]:
            (self.out / name).mkdir(exist_ok=True)
        self.rows = []
        f = self.out / "receipts/results.jsonl"
        if f.exists():
            self.rows = [json.loads(x) for x in f.read_text().splitlines()]
        self.fp = self.meta["fingerprint"]
        self.started = time.monotonic()
        self.gpu_seconds = 0
        self.engine_process = None
        self.container = None
        self.deadline = self.started + (1800 if smoke else self.plan["budget_gpu_hours"] * 3600)
        for file, digest in self.plan["input_hashes"].items():
            if sha((ROOT / file).read_bytes()) != digest:
                raise ValueError("frozen input drift: " + file)

    def log(self, event, **fields):
        with (self.out / "receipts/run.log").open("a") as f:
            stamp = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
            f.write(json.dumps(dict(event=event, at=stamp, **fields)) + "\n")

    def guard(self):
        if STOP.is_set():
            raise InterruptedError("stop requested")
        if time.monotonic() > self.deadline:
            raise TimeoutError("frozen budget reached")
        claim = json.loads(Path(self.runtime["claim_file"]).read_text())
        if (
            not claim
            or claim.get("gpu_uuid") != self.runtime["gpu_uuid"]
            or claim.get("expires_at", 0) <= time.time()
        ):
            raise InterruptedError("claim absent or expired")

    def request(self, messages, max_tokens=512):
        self.guard()
        body = {
            "model": SERVED,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            self.runtime["engine_base"] + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t = time.monotonic()
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
        elapsed = time.monotonic() - t
        usage = data.get("usage", {})
        if not usage.get("prompt_tokens") or not usage.get("completion_tokens"):
            raise ValueError("missing token counters")
        return data, elapsed

    def save(self, row):
        row.update(
            product=self.product,
            arm=self.arm,
            fingerprint=self.fp,
            phase="smoke" if self.smoke else "measurement",
        )
        with (self.out / "receipts/results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)
        write_json(
            self.out / "checkpoint.json",
            {"completed": [r.get("id") for r in self.rows], "fingerprint": self.fp},
        )

    def power(self):
        s = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                self.runtime["gpu_uuid"],
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return float(s.strip())

    @contextlib.contextmanager
    def engine(self, model, cache=False):
        self.guard()
        start = time.monotonic()
        reading = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                self.runtime["gpu_uuid"],
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if float(reading) < self.plan["minimum_free_mib"]:
            raise RuntimeError("insufficient fresh GPU headroom")
        logfile = (
            self.out / "receipts" / f"engine-vllm-{model}-{int(cache)}-{int(time.time())}.log"
        ).open("w")
        try:
            # A stale/foreign container is a hold, never removed to make room.
            if (
                subprocess.run(
                    ["docker", "inspect", CONTAINER],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            ):
                raise RuntimeError("container already exists; inspect ownership before recovery")
            image = self.plan["engines"]["vllm"]["image"]
            cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                CONTAINER,
                "--label",
                "ronanlabs.owner=w18",
                "--gpus",
                "device=" + self.runtime["gpu_uuid"],
                "--network",
                "host",
                "--ipc",
                "host",
                "-v",
                self.runtime["model_root"] + ":/weights:ro",
                "--entrypoint",
                "python3",
                image,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                "/weights/Qwen__Qwen3-" + model,
                "--served-model-name",
                SERVED,
                "--host",
                self.runtime["engine_bind"],
                "--port",
                str(self.runtime["engine_port"]),
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "8192",
                "--max-num-seqs",
                "4",
                "--max-num-batched-tokens",
                "8192",
                "--kv-cache-memory-bytes",
                str(self.plan["kv_bytes"]),
                "--block-size",
                "16",
                "--enforce-eager",
                "--seed",
                "42",
                "--enable-prefix-caching" if cache else "--no-enable-prefix-caching",
            ]
            self.log("engine-command", model=model, cache=cache, image=image)
            self.container = CONTAINER
            self.engine_process = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT)
            ready = time.monotonic() + 420
            while time.monotonic() < ready:
                self.guard()
                if self.engine_process.poll() is not None:
                    raise RuntimeError("engine exited before readiness; see engine receipt")
                try:
                    with urllib.request.urlopen(
                        self.runtime["engine_base"] + "/health", timeout=2
                    ) as r:
                        if r.status == 200:
                            break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise TimeoutError("engine readiness deadline")
            # Fixed warmup, identical to v1: two requests, excluded from scoring.
            for _ in range(2):
                self.request([{"role": "user", "content": "Reply with OK only. /no_think"}], 16)
            self.log("ready", model=model, cache=cache)
            yield
        finally:
            if self.container:
                subprocess.run(
                    ["docker", "stop", "-t", "10", self.container],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                )
                self.container = None
            if self.engine_process and self.engine_process.poll() is None:
                self.engine_process.terminate()
                try:
                    self.engine_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.engine_process.kill()
                    self.engine_process.wait(timeout=5)
            self.engine_process = None
            self.gpu_seconds += time.monotonic() - start
            logfile.close()

    def measure(self):
        inputs = [json.loads(x) for x in (ROOT / "gsm8k-32.jsonl").read_text().splitlines()]
        if self.smoke:
            inputs = [{"id": "smoke-control", "question": "What is 7 plus 8?", "answer": "#### 15"}]
        costing = self.plan["costing"]
        for model in ["4B", "8B"]:
            for cache in [False, True]:
                key = f"{model}-{int(cache)}"
                done = {r["id"] for r in self.rows}
                if all(f"{key}-{x['id']}" in done for x in inputs):
                    continue
                with self.engine(model, cache):
                    for inp in inputs:
                        rowid = f"{key}-{inp['id']}"
                        if rowid in done:
                            continue
                        msgs = [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": inp["question"]},
                        ]
                        t = time.monotonic()
                        power0 = self.power()
                        data, latency = self.request(msgs, 512)
                        choice = data["choices"][0]
                        answer = choice["message"].get("content") or ""
                        finish = choice.get("finish_reason") or "unknown"
                        elapsed = time.monotonic() - t
                        power1 = self.power()
                        # Endpoint trapezoid estimate, GPU board only; no claim of wall-plug energy.
                        energy_j = (power0 + power1) * 0.5 * elapsed
                        hardware = (
                            elapsed
                            * costing["capital_usd"]
                            / (costing["life_years"] * 365.25 * 86400 * costing["utilization"])
                        )
                        energy = energy_j / 3600000 * costing["electricity_usd_kwh"]
                        expected = normalized_number(inp["answer"])
                        extracted, rule = extract_final_number(answer, finish)
                        strict = normalized_number(answer)
                        self.save(
                            {
                                "id": rowid,
                                "model": model,
                                "cache": cache,
                                "input_id": inp["id"],
                                "correct": same_number(extracted, expected),
                                "extracted": extracted,
                                "extraction_rule": rule,
                                "correct_strict_v1": same_number(strict, expected),
                                "finish_reason": finish,
                                "valid": True,
                                "answer": answer,
                                "expected": expected,
                                "trace": [
                                    {
                                        "step": 0,
                                        "response": answer,
                                        "latency_s": latency,
                                        "usage": data["usage"],
                                    }
                                ],
                                "elapsed_s": elapsed,
                                "power_start_w": power0,
                                "power_end_w": power1,
                                "energy_j_estimate": energy_j,
                                "cost_usd": hardware + energy,
                                "hardware_usd": hardware,
                                "energy_usd": energy,
                                "usage": [data["usage"]],
                            }
                        )

    def summarize(self, rows):
        summary, verdict = {"groups": {}, "contrasts": {}}, []
        groups = {}
        for r in rows:
            groups.setdefault(f"{r['model']}-cache-{r['cache']}", {})[r["input_id"]] = r
        for group in sorted(groups):
            sub = groups[group]
            n = len(sub)
            k = sum(r["correct"] for r in sub.values())
            ks = sum(r["correct_strict_v1"] for r in sub.values())
            cost = sum(r["cost_usd"] for r in sub.values())
            ci = wilson(k, n)
            cis = wilson(ks, n)
            bounds = ratio_ci(sub, "cost_usd", "correct")
            rules = {}
            for r in sub.values():
                rules[r["extraction_rule"]] = rules.get(r["extraction_rule"], 0) + 1
            summary["groups"][group] = {
                "n": n,
                "correct": k,
                "wilson95": ci,
                "correct_strict_v1": ks,
                "wilson95_strict_v1": cis,
                "cost_usd": cost,
                "usd_per_correct": cost / k if k else None,
                "usd_per_correct_ci95": [v if math.isfinite(v) else None for v in bounds],
                "extraction_rules": rules,
                "truncated": sum(r["finish_reason"] == "length" for r in sub.values()),
                "mean_completion_tokens": statistics.mean(
                    r["usage"][0]["completion_tokens"] for r in sub.values()
                ),
            }
            verdict.append(
                f"{group}: extracted correct answers {k}/{n}, Wilson 95% CI [{ci[0]:.6f}, {ci[1]:.6f}], from receipts/results.jsonl."
            )
            verdict.append(
                f"{group}: v1 strict-format correct answers {ks}/{n}, Wilson 95% CI [{cis[0]:.6f}, {cis[1]:.6f}], from receipts/results.jsonl."
            )
            verdict.append(
                f"{group}: total modeled cost USD {cost:.8f} (n={n} attempted answers), from receipts/results.jsonl; board-energy estimates exclude CPU and wall-plug losses."
            )
            if k:
                verdict.append(
                    f"{group}: modeled USD per extracted correct answer {cost / k:.8f} (n={k} correct answers), from receipts/results.jsonl."
                )
            if all(math.isfinite(v) for v in bounds):
                verdict.append(
                    f"{group}: modeled cost per correct answer 95% bootstrap CI [{bounds[0]:.8f}, {bounds[1]:.8f}] USD (n={n} attempted answers), from receipts/results.jsonl."
                )
        pairs = [
            ("4B-vs-8B-cache-False", "4B-cache-False", "8B-cache-False"),
            ("4B-vs-8B-cache-True", "4B-cache-True", "8B-cache-True"),
            ("4B-ON-vs-OFF", "4B-cache-True", "4B-cache-False"),
            ("8B-ON-vs-OFF", "8B-cache-True", "8B-cache-False"),
        ]
        for name, a, b in pairs:
            if a not in groups or b not in groups:
                continue
            ka = summary["groups"][a]["correct"]
            kb = summary["groups"][b]["correct"]
            n = len(set(groups[a]) & set(groups[b]))
            point = (
                (summary["groups"][a]["cost_usd"] / ka) / (summary["groups"][b]["cost_usd"] / kb)
                if ka and kb
                else None
            )
            ci, undefined = paired_ratio_ci(groups[a], groups[b])
            if point is None or ci is None:
                decision = "undefined"
            elif ci[1] < 1:
                decision = f"{a} lower"
            elif ci[0] > 1:
                decision = f"{b} lower"
            else:
                decision = "no detectable difference"
            summary["contrasts"][name] = {
                "ratio": point,
                "ci95": ci,
                "undefined_draws": undefined,
                "n_paired_questions": n,
                "decision": decision,
            }
            if point is not None and ci is not None:
                verdict.append(
                    f"{name}: cost-per-correct ratio {point:.6f}, paired bootstrap 95% CI [{ci[0]:.6f}, {ci[1]:.6f}] (n={n} paired questions); decision: {decision}; from receipts/results.jsonl."
                )
            else:
                verdict.append(
                    f"{name}: ratio undefined (a zero denominator or more than 100 of 2000 undefined draws; n={n} paired questions); from receipts/results.jsonl."
                )
        return summary, verdict

    def package(self, status, error=None):
        if not self.rows:
            self.save(
                {"id": "execution-disposition", "valid": False, "status": status, "error": error}
            )
        env = dict(
            self.plan["environment"],
            fingerprint=self.fp,
            engines=self.plan["engines"],
            models=self.plan["models"],
            transport="loopback to the runner-owned engine; no router",
        )
        write_json(self.out / "receipts/environment.json", env)
        write_json(
            self.out / "receipts/outcome.json",
            {
                "status": status,
                "error": error,
                "gpu_hours": self.gpu_seconds / 3600,
                "phase": "smoke" if self.smoke else "measurement",
            },
        )
        for name in [
            "protocol-" + PRODUCT + ".md",
            "freeze.json",
            "gsm8k-32.jsonl",
            "sources.json",
        ]:
            (self.out / "receipts" / name).write_bytes((ROOT / name).read_bytes())
        rows = [r for r in self.rows if r.get("valid")]
        verdict = [
            f"Execution disposition: {status}.",
            "This package is pending PI review; smoke observations are not study estimates."
            if self.smoke
            else "Prospectively frozen pilot; no broader population or production-performance claim.",
            "All source observations are in receipts/results.jsonl and receipts/outcome.json.",
        ]
        summary = {}
        if rows:
            summary, lines = self.summarize(rows)
            verdict += lines
        if error:
            verdict.append(
                "Execution stopped; consult receipts/outcome.json before interpreting these observations."
            )
        write_json(self.out / "receipts/summary.json", summary)
        (self.out / "VERDICT.md").write_text("\n\n".join(verdict) + "\n")
        self.log("package-written", status=status)
        receipts = sorted(
            "receipts/" + f.name for f in (self.out / "receipts").iterdir() if f.is_file()
        )
        frozen = [
            {"path": "receipts/" + f, "sha256": sha((self.out / "receipts" / f).read_bytes())}
            for f in ["freeze.json", "gsm8k-32.jsonl"]
        ]
        manifest = {
            "schema_version": "1.0",
            "product_id": PRODUCT,
            "run_id": self.runid,
            "title": PRODUCT + (" smoke" if self.smoke else " prospective pilot"),
            "lane": "L3",
            "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
            "preregistration": {
                "sha256": self.meta["protocol_sha256"],
                "url": self.meta["url"],
                "timestamp_verifiable": True,
            },
            "backlog": False,
            "arms": [self.arm],
            "seeds": [42],
            "frozen_inputs": frozen,
            "environment": env,
            "receipts": receipts,
            "figures": [],
            "publication_tiers": ["P1"],
        }
        write_json(self.out / "manifest.json", manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--product", required=True, choices=[PRODUCT])
    p.add_argument("--arm", required=True, choices=["cost"])
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    run = Run(a.arm, a.smoke)
    try:
        run.measure()
        run.package("completed")
    except Exception as e:
        run.log("failed", kind=type(e).__name__)
        run.package(
            "interrupted" if isinstance(e, InterruptedError) else "failed",
            type(e).__name__ + ": " + str(e),
        )
        raise


if __name__ == "__main__":
    main()
