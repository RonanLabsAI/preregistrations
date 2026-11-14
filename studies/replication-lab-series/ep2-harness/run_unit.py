"""Queue entrypoint for replication episode 2 (runs on the worker host, outside the container).

Units: base (evaluate the untrained model), sft / cig (train with the authors' recipe,
then evaluate), compare (CPU-only paired comparison of the three arms). Training and
evaluation run in the pinned vLLM image on the worker's dedicated GPU only. Work is
resumable: training restarts from its latest checkpoint and evaluation skips finished
problems. A native K3 package is written to RL_RECEIPT_DIR when the unit completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import k3

PRODUCT = "replication-lab-series-ep2"
TITLE = "Replication lab episode 2: CIG-SFT versus matched SFT on Qwen2.5-Math-1.5B"
LOCK = HERE / "inputs.lock.json"
CODE = ["run_unit.py", "train_arm.py", "eval_math500.py", "render_prompts.py", "container_setup.sh", "inputs.lock.json"]
CONTAINER = "w16b-rep"
CLAIMED_SFT, CLAIMED_CIG, CLAIMED_BASE = 41.5, 68.4, 39.0
HALF_GAP = (CLAIMED_CIG - CLAIMED_SFT) / 2


def code_sha() -> str:
    return hashlib.sha256(b"".join((HERE / f).read_bytes() for f in CODE) +
                          (HERE.parent / "common" / "k3.py").read_bytes()).hexdigest()[:12]


def fingerprint(work: Path, lock: dict) -> str:
    tr = json.loads((work / "pydeps/train-env.json").read_text())
    ev = json.loads((work / "pydeps/eval-env.json").read_text())
    return (f"rtxpro6000-sm120|image@{lock['image_digest'][7:19]}|torch{tr['torch']}|tf{tr['transformers']}"
            f"|trl{tr['trl']}|vllm{ev['vllm']}|mathverify{ev['math-verify']}|cig@{lock['cig_commit'][:7]}|code@{code_sha()}")


def verify_inputs(work: Path, lock: dict) -> None:
    for rel, digest in lock["files"].items():
        if k3.sha256_file(work / rel) != digest:
            raise SystemExit(f"KILL: input drift for {rel}")


class Docker:
    def __init__(self, lock: dict, work: Path, gpu_uuid: str):
        self.lock, self.work, self.gpu = lock, work, gpu_uuid
        self.proc = None

    def run(self, inner: list[str], log_path: Path) -> int:
        subprocess.call(["docker", "rm", "-f", CONTAINER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cmd = ["docker", "run", "--rm", "--name", CONTAINER, "--gpus", f"device={self.gpu}", "--ipc=host",
               "--network", "host", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
               "-e", "DISABLE_VERSION_CHECK=0", "-v", f"{self.work}:/work", "-v", f"{HERE}:/harness:ro",
               "--entrypoint", "bash", self.lock["image"], "-lc", " ".join(inner)]
        with open(log_path, "a") as log:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            return self.proc.wait()

    def stop(self) -> None:
        subprocess.call(["docker", "stop", "-t", "30", CONTAINER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def summarize_eval(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    per = {r["index"]: [s["correct"] for s in r["samples"]] for r in rows}
    n = len(per)
    avg = 100 * sum(sum(v) / len(v) for v in per.values()) / n
    pas = 100 * sum(any(v) for v in per.values()) / n
    trunc = sum(s["finish"] == "length" for r in rows for s in r["samples"]) / max(1, sum(len(r["samples"]) for r in rows))
    return {"n_problems": n, "avg_at_16": avg, "pass_at_16": pas, "truncated_share": trunc}


def paired_bootstrap(a: dict, b: dict, seed: int, n_boot: int = 10_000) -> dict:
    idx = sorted(set(a) & set(b))
    da = [sum(a[i]) / len(a[i]) for i in idx]
    db = [sum(b[i]) / len(b[i]) for i in idx]
    pa = [float(any(a[i])) for i in idx]
    pb = [float(any(b[i])) for i in idx]
    rng = random.Random(seed)
    n = len(idx)

    def stat(x, y, sel):
        return 100 * sum(x[k] - y[k] for k in sel) / len(sel)

    full = list(range(n))
    point_avg, point_pass = stat(db, da, full), stat(pb, pa, full)
    boots_avg, boots_pass = [], []
    for _ in range(n_boot):
        sel = [rng.randrange(n) for _ in range(n)]
        boots_avg.append(stat(db, da, sel))
        boots_pass.append(stat(pb, pa, sel))
    boots_avg.sort()
    boots_pass.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {"n_problems": n, "avg_diff": point_avg, "avg_ci": [boots_avg[lo], boots_avg[hi]],
            "pass_diff": point_pass, "pass_ci": [boots_pass[lo], boots_pass[hi]], "n_boot": n_boot, "seed": seed}


def classify(cmp: dict) -> str:
    lo = cmp["avg_ci"][0]
    if lo > 0 and cmp["avg_diff"] >= HALF_GAP:
        return "supported"
    if lo > 0:
        return "direction only"
    return "not supported"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["base", "sft", "cig", "compare"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--gpu-uuid", required=True)
    ap.add_argument("--prereg", type=Path, default=HERE / "PREREG.md")
    ap.add_argument("--prereg-url")
    ap.add_argument("--expect-fingerprint")
    ap.add_argument("--print-fingerprint", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="5 steps / 5 problems; never a scored result")
    ap.add_argument("--smoke-tag", default="", help="suffix that keeps each smoke in fresh directories")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    lock = json.loads(LOCK.read_text())
    fp = fingerprint(args.work, lock)
    if args.print_fingerprint:
        print(fp)
        return 0
    k3.check_fingerprint(fp, args.expect_fingerprint)
    verify_inputs(args.work, lock)
    ident = k3.identity_from_env(PRODUCT)
    out = args.out or (Path(ident["receipt_dir"]) if ident["receipt_dir"] else None)
    if out is None or not args.prereg_url:
        ap.error("--prereg-url and --out (or RL_RECEIPT_DIR) are required")
    smoke_sfx = f"-smoke{args.smoke_tag}" if args.smoke else ""
    tag = f"{args.arm}-s{args.seed}{smoke_sfx}"
    run_dir = args.work / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "unit.log"
    docker = Docker(lock, args.work, args.gpu_uuid)

    def on_term(*_):
        docker.stop()
        sys.exit(75)

    signal.signal(signal.SIGTERM, on_term)
    with open(log_path, "a") as log:
        log.write(json.dumps({"t": k3.now_pt(), "event": "unit-start", "arm": args.arm, "run_id": ident["run_id"], "fingerprint": fp}) + "\n")

    eval_file = run_dir / "eval-math500.jsonl"
    if args.arm in ("sft", "cig"):
        steps = ["--max-steps 5 --save-steps 2"] if args.smoke else []
        rc = docker.run(["/work/venv-train/bin/python", "/harness/train_arm.py", "--arm", args.arm, "--seed", str(args.seed),
                         "--out", f"/work/runs/{tag}/model", *steps], log_path)
        if rc != 0:
            return rc
        model = f"/work/runs/{tag}/model"
    else:
        model = "/work/models/Qwen2.5-Math-1.5B"
    if args.arm != "compare":
        limit = ["--limit", "5"] if args.smoke else []
        rc = docker.run(["/work/venv-eval/bin/python", "/harness/eval_math500.py", "--model", model,
                         "--prompts", "/work/data/math500-prompts.jsonl", "--out", f"/work/runs/{tag}/eval-math500.jsonl",
                         "--seed", str(args.seed), *limit], log_path)
        if rc != 0:
            return rc

    pkg = k3.Package(out)
    pkg.log("start", arm=args.arm, seed=args.seed, fingerprint=fp, smoke=args.smoke)
    for rel, src in lock["package_inputs"].items():
        pkg.add_frozen_input(f"receipts/inputs/{rel}", args.work / src, lock["files"][src])
    pkg.add_receipt("receipts/inputs.lock.json", LOCK.read_bytes())
    pkg.add_receipt("receipts/unit.log", log_path.read_bytes())
    rows: list[dict] = []
    if args.arm == "compare":
        arms = {}
        for arm in ("base", "sft", "cig"):
            f = args.work / "runs" / f"{arm}-s{args.seed}{smoke_sfx}" / "eval-math500.jsonl"
            if not f.exists():
                raise SystemExit(f"KILL: missing {arm} evaluation")
            arms[arm] = {json.loads(line)["index"]: [s["correct"] for s in json.loads(line)["samples"]]
                         for line in f.read_text().splitlines() if line.strip()}
            pkg.add_receipt(f"receipts/eval-{arm}.jsonl", f.read_bytes())
            rows.append({"kind": "arm", "arm": arm, **summarize_eval(f)})
        cmp = paired_bootstrap(arms["sft"], arms["cig"], args.seed)
        cmp["classification"] = classify(cmp)
        rows.append({"kind": "comparison", "pair": "cig_minus_sft", **cmp})
        verdict = verdict_text(rows, cmp, args.prereg_url)
    else:
        summ = summarize_eval(eval_file)
        rows.append({"kind": "arm", "arm": args.arm, "smoke": args.smoke, **summ})
        rows += [{"kind": "problem", **json.loads(line)} for line in eval_file.read_text().splitlines() if line.strip()]
        state = run_dir / "model" / "trainer_state.json"
        if state.exists():
            pkg.add_receipt("receipts/trainer_state.json", state.read_bytes())
        verdict = arm_verdict(args.arm, summ, args.prereg_url, args.smoke)
    pkg.write_jsonl("receipts/results.jsonl", rows)
    env = {"hardware": "NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB), dedicated research worker",
           "engine": f"{lock['image']} (train: transformers + CIG-SFT fork; eval: vLLM + math-verify)", "fingerprint": fp}
    pkg.finish(identity=ident, title=TITLE, lane="L3", prereg_path=args.prereg, prereg_url=args.prereg_url,
               arm=args.arm, seed=args.seed, environment=env, verdict=verdict, tiers=["P1", "P3"])
    print(json.dumps({"ok": True, "arm": args.arm, "out": str(out)}))
    return 0


def arm_verdict(arm: str, s: dict, url: str, smoke: bool) -> str:
    claimed = {"base": CLAIMED_BASE, "sft": CLAIMED_SFT, "cig": CLAIMED_CIG}[arm]
    head = "SMOKE ONLY, not a result" if smoke else f"arm `{arm}`"
    return "\n".join([
        f"# Replication episode two: {head}", "", f"Preregistration: {url}", "",
        (f"Avg@16 for this arm was {s['avg_at_16']:.1f} points (n={s['n_problems']} problems) "
         f"against the paper's {claimed:.1f}, from receipts/results.jsonl."),
        f"Pass@16 for this arm was {s['pass_at_16']:.1f} points (n={s['n_problems']} problems), from receipts/results.jsonl.",
        (f"The share of samples stopped at the length cap was {100 * s['truncated_share']:.1f}% "
         f"(n={16 * s['n_problems']} samples), from receipts/results.jsonl."),
        "", "The arm comparison and its verdict are in the `compare` unit's package.", ""])


def verdict_text(rows: list[dict], cmp: dict, url: str) -> str:
    arms = {r["arm"]: r for r in rows if r["kind"] == "arm"}
    lines = ["# Verdict: CIG-SFT versus matched SFT (replication episode two)", "", f"Preregistration: {url}", "",
             f"Classification under the preregistered rule: **{cmp['classification']}**.",
             (f"CIG minus SFT Avg@16 was {cmp['avg_diff']:.2f} points (paired bootstrap 95% CI [{cmp['avg_ci'][0]:.2f}, "
              f"{cmp['avg_ci'][1]:.2f}], n={cmp['n_problems']} problems) against the paper's {CLAIMED_CIG - CLAIMED_SFT:.1f}, "
              "from receipts/results.jsonl."),
             (f"CIG minus SFT Pass@16 was {cmp['pass_diff']:.2f} points (95% CI [{cmp['pass_ci'][0]:.2f}, "
              f"{cmp['pass_ci'][1]:.2f}], n={cmp['n_problems']} problems), from receipts/results.jsonl.")]
    for arm, claimed in (("base", CLAIMED_BASE), ("sft", CLAIMED_SFT), ("cig", CLAIMED_CIG)):
        a = arms[arm]
        lines.append(f"Arm {arm} Avg@16 was {a['avg_at_16']:.1f} points (n={a['n_problems']} problems) against the paper's "
                     f"{claimed:.1f}, from receipts/results.jsonl.")
    lines += ["", "## What this does and does not establish", "",
              ("It tests the authors' released code and recipe at one seed on the frozen evaluation set with a fixed prompt and grader. "
               "It does not test the other five benchmarks, other seeds, the larger-model or code results, or a separately tuned "
               "SFT baseline."), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
