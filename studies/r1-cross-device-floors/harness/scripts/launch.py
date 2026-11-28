"""R1 plan generator + docker launcher. Commands are printable (--dry-run) and testable.

Device hosts/UUIDs/image come from environment at run time, never committed:
  R1_GPU_UUID_GB300 / R1_GPU_UUID_RTX6000 / R1_GPU_UUID_GB10 / R1_GPU_UUID_RTX5080
  R1_HOST_<DEVICE>, R1_WORKTREE, R1_MODEL_DIR (optional overrides); image is always the pin
Plan: configs/r1/plan.json (every run) + configs/r1/runs.json (analysis manifest).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEVICES = {"gb300": {"stratum": "provisional"}, "rtx6000": {"stratum": "provisional"},
           "gb10": {"stratum": "provisional"}, "rtx5080": {"stratum": "provisional"}}
# Declared differences from Rane (2611.29663) for the matched arm (D). Rane values are
# stated only where the card states them: prefix caching off, prefill budget 8192,
# 128 sequences, CUDA graphs on, bf16 KV; otherwise None (not stated in the card).
RANE_DECLARED_DIFFERENCES = [
    {"name": "max_num_seqs", "r1": 32, "rane": 128},
    {"name": "max_num_batched_tokens", "r1": 8192, "rane": 8192},
    {"name": "num_gpu_blocks_override", "r1": 2048, "rane": None},
    {"name": "block_size", "r1": 16, "rane": None},
    {"name": "prefix_caching", "r1": False, "rane": False},
    {"name": "chunked_prefill", "r1": True, "rane": None},
    {"name": "cuda_graphs", "r1": True, "rane": True},
    {"name": "kv_cache_dtype", "r1": "bfloat16", "rane": "bfloat16"},
]
Q4B = "qwen3-4b-instruct-2507"
Q17 = "qwen3-1.7b"
TRANSFER_ITEMS = "configs/r1/items/transfer_core.jsonl"
FULL_ITEMS = "configs/r1/items/full_dev.jsonl"
WARMUP_IDS_FILE = "configs/r1/smoke_items.txt"  # existing 100-id smoke subset; filtered at run time (no new items file)
RANE_CAPS = {"gsm8k": 768, "math500": 4096, "default": 1024}

def _run(run_id, arm, device, model, mode, kind, cache="warm", order_seed=0, passes=1,
         items=TRANSFER_ITEMS, **kw):
    r = {"run_id": run_id, "arm": arm, "device": device, "stratum": DEVICES[device]["stratum"],
         "model": model, "mode": mode, "kind": kind, "cache": cache, "order_seed": order_seed,
         "passes": passes, "items": items, "max_tokens": 4096,
         "max_num_seqs": 32, "max_num_batched_tokens": 8192}
    r.update(kw)
    return r

def build_plan() -> dict:
    runs = []
    # Transfer arm: per device x mode: 5 restarts (2 cold, 3 warm) + 1 same-process (2 passes) + 2 order
    for dev in DEVICES:
        for mode in ["default", "bi"]:
            for i in range(5):
                cache = "cold" if i < 2 else "warm"
                runs.append(_run(f"r1-transfer-{dev}-{mode}-restart{i}-{cache}", "transfer", dev, Q4B, mode,
                                 "restart", cache=cache, restart_index=i))
            runs.append(_run(f"r1-transfer-{dev}-{mode}-sameprocess", "transfer", dev, Q4B, mode,
                             "same_process", cache="warm", passes=2))
            for j, seed in enumerate([20261129, 20261130]):
                runs.append(_run(f"r1-transfer-{dev}-{mode}-order{j}", "transfer", dev, Q4B, mode,
                                 "order", cache="warm", order_seed=seed))
    # Matched Rane arm: 3 devices, default, 3 warm restarts, Rane caps
    for dev in ["rtx5080", "rtx6000", "gb300"]:
        for i in range(3):
            runs.append(_run(f"r1-rane-{dev}-restart{i}-warm", "matched_rane", dev, Q4B, "default",
                             "restart", cache="warm", restart_index=i, caps=dict(RANE_CAPS),
                             declared_differences_from_rane=[dict(d) for d in RANE_DECLARED_DIFFERENCES]))
    # Serving-path screen: GB300, Qwen3-1.7B, default, 3 warm restarts per factor
    factors = [("reference_http96", {"path": "http", "concurrency": 96}),
               ("http_concurrency_1", {"path": "http", "concurrency": 1}),
               ("offline", {"path": "offline"}),
               ("prefix_caching_on", {"path": "http", "concurrency": 96, "prefix_caching": True}),
               ("budget_768", {"path": "http", "concurrency": 96, "max_tokens": 768}),
               ("budget_4096", {"path": "http", "concurrency": 96, "max_tokens": 4096}),
               ("fp8_kv", {"path": "http", "concurrency": 96, "kv_cache_dtype": "fp8",
                           "fp8_gate": "accuracy within 2 points of bf16 before flips are read"})]
    for factor, kw in factors:
        for i in range(3):
            runs.append(_run(f"r1-screen-gb300-{factor}-restart{i}", "serving_screen", "gb300", Q17,
                             "default", "restart", cache="warm", restart_index=i, factor=factor, **kw))
    # Adapter arm: RTX PRO 6000, Qwen3-1.7B, BI, 3 warm restarts per path
    for factor in ["merged", "single_lora", "interleaved_same", "interleaved_mixed"]:
        for i in range(3):
            runs.append(_run(f"r1-adapter-rtx6000-{factor}-restart{i}", "adapter", "rtx6000", Q17,
                             "bi", "restart", cache="warm", restart_index=i, factor=factor,
                             max_loras=3 if "interleaved" in factor else 1))
    # Cost arm: GB300 + RTX PRO 6000, Qwen3-4B, BI on vs off
    for dev in ["gb300", "rtx6000"]:
        for mode in ["default", "bi"]:
            runs.append(_run(f"r1-cost-{dev}-{mode}", "cost", dev, Q4B, mode, "cost",
                             cache="warm", concurrency=32, duration_s=600, latency_n=50))
    # Logprob smoke: 100 items, logprobs off vs top-20 (byte-identity gate before margins)
    for lp in [0, 20]:
        runs.append(_run(f"r1-logprob-gb300-lp{lp}", "logprob_smoke", "gb300", Q4B, "default",
                         "smoke", cache="warm", logprobs=lp, limit=100))
    # Full set: GB300 only
    for mode in ["default", "bi"]:
        runs.append(_run(f"r1-full-gb300-{mode}-restart0-warm", "full_set", "gb300", Q4B, mode,
                         "restart", cache="warm", restart_index=0, items=FULL_ITEMS))
    # Warmup (Deviation 4 #1): one uncounted warmup per device x model x mode that uses a
    # warm cache dir; runs first in each device's queue (prepended, plan order preserved).
    # Items: the 100 smoke ids filtered from transfer_core at run time via --ids-file.
    triples = sorted({(r["device"], r["model"], r["mode"]) for r in runs if r.get("cache") == "warm"})
    warmups = [_run(f"r1-warmup-{dev}-{model}-{mode}", "warmup", dev, model, mode,
                    "warmup", cache="warm", ids_file=WARMUP_IDS_FILE)
               for dev, model, mode in triples]
    runs = warmups + runs
    return {"version": 1, "items": {"transfer_core": TRANSFER_ITEMS, "full_dev": FULL_ITEMS},
            "devices": DEVICES, "runs": runs,
            "counts": {"total_runs": len(runs), "warmup_runs": len(warmups),
                       "transfer_invocations": len([r for r in runs if r["arm"] == "transfer"])}}

def runs_manifest(plan=None) -> dict:
    plan = plan or build_plan()
    return {"runs": [{"run_id": r["run_id"], "device": r["device"], "model": r["model"],
                      "mode": r["mode"], "cache": r["cache"], "order_seed": r["order_seed"],
                      "pass": r["passes"], "stratum": r["stratum"], "arm": r["arm"],
                      "kind": r["kind"], "step": "r1", "factor": r.get("factor"),
                      "items": r.get("items"), "ids_file": r.get("ids_file")}
                     for r in plan["runs"]]}

def _env(name):
    v = os.environ.get(name)
    if not v:
        raise KeyError(f"{name} is not set (device hosts/UUIDs come from the environment, never committed)")
    return v

MODEL_PATHS = {Q4B: "/models/hf/Qwen__Qwen3-4B-Instruct-2507", Q17: "/models/hf/Qwen__Qwen3-1.7B"}
CHAT_KWARGS = {Q4B: {}, Q17: {"enable_thinking": False}}  # Qwen3-4B-Instruct-2507 has no thinking mode
PIN_IMAGE = "vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2"  # configs/r1/pin.md
ATTENTION_BACKEND = "FLASH_ATTN"  # what vLLM selects on the RTX PRO 6000 at the pin; forced on every device
NUM_GPU_BLOCKS = 2048  # 32,768 KV tokens: fits a 16 GB RTX 5080 with Qwen3-4B; identical on every device
MAX_MODEL_LEN = 8192


def docker_run_cmd(device, inner_cmd, dry_run=False, image_digest=None):
    """docker run for one run on `device`; UUID from env, image is always the pin (no override).
    The entrypoint is replaced by bash so the inner command (with its log redirect) runs as given."""
    uuid = _env(f"R1_GPU_UUID_{device.upper()}")
    image = PIN_IMAGE
    digest = image_digest if image_digest is not None else os.environ.get("R1_IMAGE_DIGEST", "")
    wt = os.environ.get("R1_WORKTREE", "$HOME/mom-lab/r1")
    cmd = ["docker", "run", "--rm", "--gpus", f"device={uuid}", "-e", f"CUDA_VISIBLE_DEVICES={uuid}", "--ipc=host",
           "-u", "1000:1000", "-e", "HOME=/tmp", "-e", "HF_HUB_OFFLINE=1",
           "-e", "NVIDIA_TF32_OVERRIDE=0"]
    if digest:
        cmd += ["-e", f"R1_IMAGE_DIGEST={digest}"]
    cmd += [
           "-v", f"{wt}:/r1", "-v", "/models/hf:/models/hf:ro", "-w", "/r1", "--entrypoint", "bash", image,
           "-c", inner_cmd]
    if dry_run:
        print(" ".join(cmd))
    return cmd


def offline_cmd(run: dict) -> list[str]:
    out = f"results/r1/{run['run_id']}"
    cache_dir = f"/r1/results/r1-cache/{run['device']}-{run['model']}-{run['mode']}"  # warm cache per device x model x mode
    cmd = ["python3", "-u", "scripts/r1/r1_offline.py", "--model", MODEL_PATHS[run["model"]],
           "--items", run["items"], "--out", out, "--run-id", run["run_id"],
           "--mode", run["mode"], "--passes", str(run["passes"]), "--order-seed", str(run["order_seed"]),
           "--max-tokens", str(run.get("max_tokens", 4096)),
           "--max-num-seqs", str(run.get("max_num_seqs", 32)),
           "--max-num-batched-tokens", str(run.get("max_num_batched_tokens", 8192)),
           "--max-model-len", str(MAX_MODEL_LEN), "--num-gpu-blocks", str(NUM_GPU_BLOCKS),
           "--attention-backend", ATTENTION_BACKEND,
           "--cache", run.get("cache", "warm"), "--cache-dir", cache_dir,
           "--chat-template-kwargs", "'" + json.dumps(CHAT_KWARGS[run["model"]]) + "'"]
    if run.get("caps"):
        cmd += ["--caps-json", "'" + json.dumps(run["caps"]) + "'"]
    if run.get("logprobs"):
        cmd += ["--logprobs", str(run["logprobs"])]
    if run.get("ids_file"):
        cmd += ["--ids-file", run["ids_file"]]
    if run.get("kind") == "warmup":
        cmd += ["--warmup"]
    return cmd


def offline_inner(run: dict) -> str:
    """Shell string run inside the container: make the run dir, run, keep the engine log (parsed into run_meta)."""
    out = f"results/r1/{run['run_id']}"
    return f"mkdir -p {out} && " + " ".join(offline_cmd(run)) + f" > {out}/engine.log 2>&1"


def shell_cmd(cmd: list[str]) -> str:
    """One shell string for ssh: every token quoted except tokens containing $HOME (expanded on the device)."""
    import shlex
    return " ".join(x if "$HOME" in x else shlex.quote(x) for x in cmd)


def write_plan(out_dir=None) -> dict:
    out = Path(out_dir) if out_dir else ROOT / "configs" / "r1"
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (out / "runs.json").write_text(json.dumps(runs_manifest(plan), indent=2), encoding="utf-8")
    return plan

def main(argv=None):
    ap = argparse.ArgumentParser(prog="launch.py")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan", action="store_true", help="write configs/r1/plan.json + runs.json")
    ap.add_argument("--device", default="gb300")
    ap.add_argument("--run-id", default="")
    a = ap.parse_args(argv)
    if a.plan:
        plan = write_plan()
        print(f"plan: {plan['counts']}")
    if a.run_id:
        plan = build_plan()
        run = next((r for r in plan["runs"] if r["run_id"] == a.run_id), None)
        if run is None:
            print(f"unknown run-id {a.run_id}", file=sys.stderr); return 1
        inner = offline_inner(run) if run["kind"] != "cost" else \
            f"python3 scripts/r1/r1_cost.py --run-id {run['run_id']}"
        cmd = docker_run_cmd(a.device or run["device"], inner, dry_run=a.dry_run)
        if not a.dry_run:
            print(shell_cmd(cmd))
    return 0

if __name__ == "__main__":
    sys.exit(main())
