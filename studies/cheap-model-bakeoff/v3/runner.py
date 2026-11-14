"""Frozen public study runner. Deployment details come only from a private runtime file.

All inference uses a scoped controller proxy forwarding named research routes.
Only the safe arithmetic tool is executable; model-generated code is never evaluated.
"""

import argparse
import ast
import concurrent.futures
import contextlib
import hashlib
import json
import math
import operator
import os
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


def bootstrap(values):
    import random

    if not values:
        return None
    rng = random.Random(42)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return [draws[49], draws[1949]]


def calculate(text):
    if len(text) > 200:
        raise ValueError("expression too long")
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def visit(n, depth=0):
        if depth > 20:
            raise ValueError("expression too deep")
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            v = n.value
        elif isinstance(n, ast.BinOp) and type(n.op) in ops:
            v = ops[type(n.op)](visit(n.left, depth + 1), visit(n.right, depth + 1))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = -visit(n.operand, depth + 1)
        else:
            raise ValueError("only numeric + - * / allowed")
        if not math.isfinite(v) or abs(v) > 1e15:
            raise ValueError("numeric bound")
        return v

    return visit(ast.parse(text, mode="eval").body)


def normalized_number(text):
    text = text.split("####")[-1].strip().replace(",", "").replace("$", "")
    m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    return float(text) if m else None


class Run:
    def __init__(self, product, arm, smoke=False):
        self.product, self.arm, self.smoke = product, arm, smoke
        self.runtime = json.loads(Path(os.environ["W16_RUNTIME"]).read_text())
        self.plan = json.loads((ROOT / "freeze.json").read_text())
        self.meta = json.loads((ROOT / "publication.json").read_text())[product]
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
        self.deadline = self.started + (1800 if smoke else self.plan["budgets"][product] * 3600)
        for file, digest in self.plan["input_hashes"].items():
            if sha((ROOT / file).read_bytes()) != digest:
                raise ValueError("frozen input drift: " + file)

    def log(self, event, **fields):
        with (self.out / "receipts/run.log").open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        event=event,
                        at=datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
                        **fields,
                    )
                )
                + "\n"
            )

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

    def request(self, messages, route, max_tokens=512):
        self.guard()
        body = {
            "model": route,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body["extra_body"] = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if route == self.runtime["qwen_route"]
            else {"options": {"num_ctx": 65536, "num_predict": 1024}}
        )
        # LiteLLM accepts provider-specific fields at the top level.
        body.update(body.pop("extra_body"))
        token = Path(self.runtime["proxy_token_file"]).read_text().strip()
        req = urllib.request.Request(
            self.runtime["proxy_base"] + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
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
    def engine(self, model, engine="vllm", cache=False):
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
            self.out / "receipts" / f"engine-{engine}-{model}-{int(cache)}-{int(time.time())}.log"
        ).open("w")
        try:
            if engine == "ollama":
                env = os.environ.copy()
                env.update(self.runtime["ollama_env"])
                env["CUDA_VISIBLE_DEVICES"] = self.runtime["gpu_uuid"]
                self.engine_process = subprocess.Popen(
                    [self.runtime["ollama_binary"], "serve"],
                    env=env,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                )
                health = self.runtime["ollama_health"]
            else:
                container_name = "w16a-engine"
                # A stale/foreign container is a hold, never removed to make room.
                if (
                    subprocess.run(
                        ["docker", "inspect", container_name],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    ).returncode
                    == 0
                ):
                    raise RuntimeError(
                        "container already exists; inspect ownership before recovery"
                    )
                image = self.plan["engines"][engine]["image"]
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    container_name,
                    "--label",
                    "ronanlabs.owner=w16a",
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
                ]
                model_path = "/weights/Qwen__Qwen3-" + model
                if engine == "vllm":
                    cmd += [
                        "-m",
                        "vllm.entrypoints.openai.api_server",
                        "--model",
                        model_path,
                        "--served-model-name",
                        "w16a",
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
                else:
                    cmd += [
                        "-m",
                        "sglang.launch_server",
                        "--model-path",
                        model_path,
                        "--served-model-name",
                        "w16a",
                        "--host",
                        self.runtime["engine_bind"],
                        "--port",
                        str(self.runtime["engine_port"]),
                        "--dtype",
                        "bfloat16",
                        "--context-length",
                        "8192",
                        "--max-running-requests",
                        "4",
                        "--max-prefill-tokens",
                        "8192",
                        "--chunked-prefill-size",
                        "8192",
                        "--max-total-tokens",
                        str(self.plan["kv_tokens"]),
                        "--page-size",
                        "16",
                        "--mem-fraction-static",
                        "0.5",
                        "--disable-cuda-graph",
                        "--random-seed",
                        "42",
                    ] + ([] if cache else ["--disable-radix-cache"])
                # Equal token capacity and BF16 KV in both engines; derived from frozen model config.
                self.log("engine-command", engine=engine, model=model, cache=cache, image=image)
                self.container = container_name
                self.engine_process = subprocess.Popen(
                    cmd, stdout=logfile, stderr=subprocess.STDOUT
                )
                health = self.runtime["engine_health"]
            ready = time.monotonic() + 420
            while time.monotonic() < ready:
                self.guard()
                if self.engine_process.poll() is not None:
                    raise RuntimeError("engine exited before readiness; see engine receipt")
                try:
                    with urllib.request.urlopen(health, timeout=2) as r:
                        if r.status == 200:
                            break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise TimeoutError("engine readiness deadline")
            # Fixed warmup: two requests; exact same chat shape for all Qwen arms.
            route = (
                self.runtime["needle_route"] if engine == "ollama" else self.runtime["qwen_route"]
            )
            for _ in range(2):
                self.request(
                    [{"role": "user", "content": "Reply with OK only. /no_think"}],
                    route,
                    16 if engine != "ollama" else 1024,
                )
            self.log("ready", engine=engine, model=model, cache=cache)
            yield route
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

    def needle(self):
        inputs = [json.loads(x) for x in (ROOT / "needle.jsonl").read_text().splitlines()]
        if self.smoke:
            inputs = [
                dict(
                    inputs[0],
                    id="smoke-control",
                    prompt="The secret code is SMOKE16A. Return only the secret code.",
                    expected="SMOKE16A",
                    expected_prompt_tokens=15,
                )
            ]
        with self.engine("nemotron", "ollama") as route:
            done = {r["id"] for r in self.rows}
            for inp in inputs:
                if inp["id"] in done:
                    continue
                data, elapsed = self.request(
                    [{"role": "user", "content": inp["prompt"]}], route, 1024
                )
                msg = data["choices"][0]
                answer = msg["message"].get("content") or ""
                count = data["usage"]["prompt_tokens"]
                expected = inp["expected_prompt_tokens"]
                valid = count >= expected * 0.95 and count + 1024 <= 65536
                self.save(
                    {
                        "id": inp["id"],
                        "target_tokens": inp.get("target_tokens"),
                        "depth": inp.get("depth"),
                        "seed": inp.get("seed"),
                        "prompt_sha256": sha(inp["prompt"].encode()),
                        "expected_prompt_tokens": expected,
                        "prompt_eval_count": count,
                        "counter_source": "LiteLLM usage.prompt_tokens mapped from Ollama prompt_eval_count",
                        "serving_context_tokens": 65536,
                        "correct": answer.strip() == inp["expected"],
                        "valid": valid,
                        "answer": answer,
                        "usage": data["usage"],
                        "elapsed_s": elapsed,
                        "finish_reason": msg.get("finish_reason"),
                    }
                )
                if not valid:
                    raise ValueError("context/counter gate failed")

    def prefix(self):
        engine = self.arm
        prefix = " ".join(
            [
                "The shared public reference describes a blue triangle, a green square, and a red circle."
            ]
            * 128
        )
        repeats = 1 if self.smoke else 5
        for cache in [False, True]:
            with self.engine("4B", engine, cache) as route:
                for rep in range(repeats):
                    # Disjoint rep prefix avoids cross-repetition cache warming.
                    context = (
                        f"Smoke group {rep}. " if self.smoke else f"Experiment group {rep}. "
                    ) + prefix
                    for phase in ["cold", "warm"]:
                        msgs = [
                            [
                                {
                                    "role": "user",
                                    "content": context
                                    + f"\nQuestion {j}: What color is the triangle? Reply with just the color. /no_think",
                                }
                            ]
                            for j in range(4)
                        ]
                        t = time.monotonic()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                            results = list(pool.map(lambda m: self.request(m, route, 16), msgs))
                        wall = time.monotonic() - t
                        for j, (data, elapsed) in enumerate(results):
                            self.save(
                                {
                                    "id": f"{cache}-{rep}-{phase}-{j}",
                                    "engine": engine,
                                    "cache": cache,
                                    "rep": rep,
                                    "cache_phase": phase,
                                    "batch_size": 4,
                                    "batch_wall_s": wall,
                                    "elapsed_s": elapsed,
                                    "usage": data["usage"],
                                    "answer": data["choices"][0]["message"].get("content"),
                                    "valid": True,
                                }
                            )

    def quality(self):
        inputs = [json.loads(x) for x in (ROOT / "gsm8k-32.jsonl").read_text().splitlines()]
        if self.product == "cheap-model-bakeoff":
            inputs += [json.loads(x) for x in (ROOT / "bbh-16.jsonl").read_text().splitlines()]
        if self.smoke:
            inputs = [{"id": "smoke-control", "question": "What is 7 plus 8?", "answer": "#### 15"}]
        for model in ["4B", "8B"]:
            for cache in [False, True] if self.product == "cost-per-answer-shootout" else [False]:
                key = f"{model}-{int(cache)}"
                done = {r["id"] for r in self.rows}
                if all(f"{key}-{x['id']}" in done for x in inputs):
                    continue
                with self.engine(model, "vllm", cache) as route:
                    for inp in inputs:
                        rowid = f"{key}-{inp['id']}"
                        if rowid in done:
                            continue
                        system = (
                            "Answer the math problem. Return only the final number. /no_think"
                            if self.product == "cost-per-answer-shootout"
                            else 'You are a bounded calculator agent. Reply with exactly one JSON object: {"tool":"calculator","expression":"numeric expression"} to use the arithmetic tool, or {"answer":"final number"}. At most three tool calls. Only + - * / and parentheses are supported. /no_think'
                        )
                        msgs = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": inp["question"]},
                        ]
                        t = time.monotonic()
                        power0 = self.power()
                        trace = []
                        answer = ""
                        usage = []
                        for step in range(1 if self.product == "cost-per-answer-shootout" else 4):
                            data, latency = self.request(msgs, route, 512)
                            usage.append(data["usage"])
                            answer = data["choices"][0]["message"].get("content") or ""
                            trace.append(
                                {
                                    "step": step,
                                    "response": answer,
                                    "latency_s": latency,
                                    "usage": data["usage"],
                                }
                            )
                            if self.product == "cost-per-answer-shootout":
                                break
                            try:
                                parsed = json.loads(answer)
                                if "answer" in parsed:
                                    answer = str(parsed["answer"])
                                    break
                                if parsed.get("tool") != "calculator" or step == 3:
                                    break
                                result = calculate(parsed["expression"])
                                msgs.extend(
                                    [
                                        {"role": "assistant", "content": answer},
                                        {
                                            "role": "user",
                                            "content": json.dumps({"tool_result": result}),
                                        },
                                    ]
                                )
                            except (
                                ValueError,
                                KeyError,
                                TypeError,
                                ZeroDivisionError,
                                SyntaxError,
                            ) as e:
                                trace[-1]["tool_error"] = type(e).__name__
                                break
                        elapsed = time.monotonic() - t
                        power1 = self.power()
                        # Endpoint trapezoid estimate, GPU board only; no claim of wall-plug energy.
                        energy_j = (power0 + power1) * 0.5 * elapsed
                        hardware = (
                            elapsed
                            * self.plan["costing"]["capital_usd"]
                            / (
                                self.plan["costing"]["life_years"]
                                * 365.25
                                * 86400
                                * self.plan["costing"]["utilization"]
                            )
                        )
                        energy = energy_j / 3600000 * self.plan["costing"]["electricity_usd_kwh"]
                        expected = normalized_number(inp["answer"])
                        pred = normalized_number(answer)
                        self.save(
                            {
                                "id": rowid,
                                "model": model,
                                "cache": cache,
                                "input_id": inp["id"],
                                "correct": pred is not None and pred == expected,
                                "valid": True,
                                "answer": answer,
                                "expected": expected,
                                "trace": trace,
                                "elapsed_s": elapsed,
                                "power_start_w": power0,
                                "power_end_w": power1,
                                "energy_j_estimate": energy_j,
                                "cost_usd": hardware + energy,
                                "hardware_usd": hardware,
                                "energy_usd": energy,
                                "usage": usage,
                            }
                        )

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
            "protocol-" + self.product + ".md",
            "freeze.json",
            "gsm8k-32.jsonl",
            "bbh-16.jsonl",
            "sources.json",
        ] + (["needle.jsonl"] if "needle" in self.product else []):
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
        if "prefix" in self.product:
            for cache in [False, True]:
                vals = [
                    r["batch_wall_s"]
                    for r in rows
                    if r.get("cache") == cache
                    and r.get("cache_phase") == "warm"
                    and r["id"].endswith("-0")
                ]
                if vals:
                    ci = bootstrap(vals)
                    summary[str(cache)] = {
                        "n": len(vals),
                        "mean_batch_wall_s": statistics.mean(vals),
                        "ci95": ci,
                    }
                    verdict.append(
                        f"Cache enabled {cache}: mean warm batch wall time {statistics.mean(vals):.6f} seconds, 95% bootstrap CI [{ci[0]:.6f}, {ci[1]:.6f}] (n={len(vals)} batches), from receipts/results.jsonl."
                    )
        else:
            groups = {
                r.get("model", "nemotron") + ("-cache-" + str(r["cache"]) if "cache" in r else "")
                for r in rows
            }
            for group in sorted(groups):
                sub = [
                    r
                    for r in rows
                    if r.get("model", "nemotron")
                    + ("-cache-" + str(r["cache"]) if "cache" in r else "")
                    == group
                ]
                k = sum(r.get("correct", False) for r in sub)
                n = len(sub)
                ci = wilson(k, n)
                summary[group] = {"correct": k, "n": n, "wilson95": ci}
                verdict.append(
                    f"{group}: exact answers {k}/{n}, Wilson 95% CI [{ci[0]:.6f}, {ci[1]:.6f}], from receipts/results.jsonl."
                )
                if "cost" in self.product:
                    cost = sum(r["cost_usd"] for r in sub)
                    summary[group]["cost_usd"] = cost
                    summary[group]["usd_per_correct"] = cost / k if k else None
                    import random

                    rng = random.Random(42)
                    ratios = []
                    for _ in range(2000):
                        sample = rng.choices(sub, k=n)
                        correct = sum(r["correct"] for r in sample)
                        ratios.append(
                            sum(r["cost_usd"] for r in sample) / correct if correct else math.inf
                        )
                    ratios.sort()
                    bounds = [ratios[49], ratios[1949]]
                    summary[group]["usd_per_correct_ci95"] = [
                        v if math.isfinite(v) else None for v in bounds
                    ]
                    if all(math.isfinite(v) for v in bounds):
                        verdict.append(
                            f"{group}: modeled cost per correct answer 95% bootstrap CI [{bounds[0]:.8f}, {bounds[1]:.8f}] USD (n={n} attempted answers), from receipts/results.jsonl."
                        )

                    verdict.append(
                        f"{group}: total modeled cost USD {cost:.8f} (n={n} attempted answers), from receipts/results.jsonl; board-energy estimates exclude CPU and wall-plug losses."
                    )
                    if k:
                        verdict.append(
                            f"{group}: modeled USD per correct answer {cost / k:.8f} (n={k} correct answers), from receipts/results.jsonl."
                        )
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
            "product_id": self.product,
            "run_id": self.runid,
            "title": self.product + (" smoke" if self.smoke else " prospective pilot"),
            "lane": "L1" if "prefix" in self.product else "L3",
            "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
            "preregistration": {
                "sha256": self.meta["protocol_sha256"],
                "url": self.meta["url"],
                "timestamp_verifiable": True,
            },
            "backlog": False,
            "arms": [self.arm],
            "seeds": [42, 43] if "needle" in self.product else [42],
            "frozen_inputs": frozen,
            "environment": env,
            "receipts": receipts,
            "figures": [],
            "publication_tiers": ["P1"],
        }
        write_json(self.out / "manifest.json", manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--product", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    run = Run(a.product, a.arm, a.smoke)
    try:
        if "needle" in a.product:
            run.needle()
        elif "prefix" in a.product:
            run.prefix()
        else:
            run.quality()
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
