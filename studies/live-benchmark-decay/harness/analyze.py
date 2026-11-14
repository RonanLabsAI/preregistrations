"""Run the preregistered live-benchmark decay analysis and write a native K3 package.

Usage (Mission Control CPU executor exports RL_* identity variables):
  uv run --frozen python -s analyze.py --inputs <dir> --out "$RL_RECEIPT_DIR" \
      --prereg PREREG.md --prereg-url <https url> --seed 20261114 --expect-fingerprint <fp>
  analyze.py --print-fingerprint   # prints the observed environment fingerprint and exits
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import decay  # noqa: E402
from common import k3  # noqa: E402

PRODUCT = "live-benchmark-decay"
TITLE = "Live-benchmark decay: are scores from different release windows comparable?"
ARM = "decay"
LOCK = HERE / "inputs.lock.json"
CODE_FILES = [HERE / "decay.py", HERE / "analyze.py", HERE.parent / "common" / "k3.py", LOCK]


def fingerprint() -> str:
    import pyarrow

    lock = json.loads(LOCK.read_text())
    code = hashlib.sha256(b"".join(p.read_bytes() for p in CODE_FILES)).hexdigest()[:12]
    lcb = lock["lcb"]["revision"][:8]
    lb = lock["livebench"]["model_judgment"]["revision"][:8]
    host = k3.host_summary()
    return (f"cpu-{host['machine']}|py{host['python']}|numpy{np.__version__}|pyarrow{pyarrow.__version__}"
            f"|lcb@{lcb}|livebench@{lb}|code@{code}")


def load_inputs(inputs: Path) -> tuple[dict, list[dict], dict]:
    lock = json.loads(LOCK.read_text())
    for name, meta in lock["frozen_files"].items():
        got = k3.sha256_file(inputs / name)
        if got != meta["sha256"]:
            raise SystemExit(f"KILL: input drift for {name}: {got} != {meta['sha256']}")
    perf = json.loads((inputs / "lcb_performances_generation.json").read_text())
    with open(inputs / "livebench_model_judgment.csv", newline="") as f:
        judgments = [{**r, "turn": int(r["turn"]), "score": float(r["score"])} for r in csv.DictReader(f)]
    index = {}
    with open(inputs / "livebench_release_index.csv", newline="") as f:
        for row in csv.DictReader(f):
            index[row["question_id"]] = (row["category"], row["task"], row["livebench_release_date"])
    return perf, judgments, index


def fmt_ci(ci: list[float]) -> str:
    return f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"


def run(inputs: Path, seed: int, n_boot: int) -> dict:
    perf, judgments, index = load_inputs(inputs)
    mat, extra = decay.lcb_matrix(perf)
    windows = decay.included_windows(mat)
    if len(windows) < 3:
        raise SystemExit("KILL: fewer than three LCB windows meet the minimum problem count")
    primary = decay.analyze_corpus(mat, windows, seed, n_boot)
    primary["classification"] = decay.classify(primary["ci95"]["adj_mean_abs_drift"])
    primary["n_problems"] = int(sum(primary["window_sizes"].values()))
    primary["excluded_windows"] = sorted(set(map(str, mat.window)) - set(windows))
    contamination = decay.contamination_exploratory(mat, extra, seed)
    secondary = {}
    for cat in sorted({v[0] for v in index.values()}):
        lbm, _ = decay.livebench_matrix(judgments, index, cat)
        secondary[cat] = decay.livebench_pairs(lbm, seed, n_boot)
    unmatched = sum(1 for r in judgments if r["question_id"] not in index)
    return {"primary_lcb": primary, "secondary_livebench": secondary,
            "exploratory_lcb_release_gap": contamination,
            "livebench_judgments_total": len(judgments), "livebench_judgments_unmatched": unmatched}


def verdict_text(res: dict, prereg_url: str) -> str:
    p = res["primary_lcb"]
    pt, ci = p["point"], p["ci95"]
    lines = [
        "# Verdict: live-benchmark decay (LiveCodeBench primary, LiveBench secondary)",
        "",
        f"Preregistration: {prereg_url}",
        "",
        "## Primary (LiveCodeBench, adjacent calendar quarters, difficulty-adjusted)",
        "",
        f"Classification under the preregistered three-point tolerance: **{p['classification']}**.",
        (f"Mean absolute difficulty-adjusted drift between adjacent quarters was {pt['adj_mean_abs_drift']:.2f} points "
         f"({fmt_ci(ci['adj_mean_abs_drift'])}, n={pt['n_cells_adj']} model-pair cells), from receipts/results.jsonl."),
        (f"Unadjusted mean absolute drift was {pt['raw_mean_abs_drift']:.2f} points "
         f"({fmt_ci(ci['raw_mean_abs_drift'])}, n={pt['n_cells_raw']} model-pair cells), from receipts/results.jsonl."),
        (f"Median Kendall tau-b between adjacent-quarter model rankings was {pt['median_tau']:.3f} "
         f"({fmt_ci(ci['median_tau'])}, n={pt['pairs']} quarter pairs), from receipts/results.jsonl."),
        (f"The panel covered n={p['n_models']} models and n={p['n_problems']} problems in the included quarters, "
         "from receipts/results.jsonl."),
        "",
        "## Secondary (LiveBench, adjacent releases, task-matched)",
        "",
    ]
    for cat, pairs in res["secondary_livebench"].items():
        for r in pairs:
            if "skipped" in r:
                lines.append(f"- {cat} {r['pair'][0]} to {r['pair'][1]}: not analyzed ({r['skipped']}; "
                             f"n={r['n_models']} complete models).")
                continue
            lines.append(
                f"- {cat} {r['pair'][0]} to {r['pair'][1]}: adjusted mean absolute drift "
                f"{r['point']['adj_mean_abs_drift']:.2f} points ({fmt_ci(r['ci95_adj'])}, n={r['n_models']} models), "
                f"classified {r['classification']}, from receipts/results.jsonl.")
    lines += [
        "",
        "## Exploratory (not a preregistered test)",
        "",
        "Post-minus-pre release-marker gaps use LiveCodeBench's own leaderboard markers, not verified training cutoffs.",
    ]
    for r in res["exploratory_lcb_release_gap"]:
        lines.append(f"- {r['model']} (marker {r['release_marker']}): {r['post_minus_pre_pp']:.2f} points "
                     f"({fmt_ci(r['ci95'])}, n={r['n_after']} post-marker problems), from receipts/results.jsonl.")
    lines += [
        "",
        "## What this does and does not establish",
        "",
        "It measures score comparability across problem windows for a fixed set of models on two public live benchmarks. "
        "It does not identify why windows differ beyond the difficulty and task mix it adjusts for. "
        "It does not establish contamination. The snapshots are frozen public files; the live leaderboards may have changed since.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--prereg", type=Path, default=HERE / "PREREG.md")
    ap.add_argument("--prereg-url")
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=decay.N_BOOT)
    ap.add_argument("--expect-fingerprint")
    ap.add_argument("--print-fingerprint", action="store_true")
    args = ap.parse_args()
    fp = fingerprint()
    if args.print_fingerprint:
        print(fp)
        return 0
    k3.check_fingerprint(fp, args.expect_fingerprint)
    ident = k3.identity_from_env(PRODUCT)
    out = args.out or (Path(ident["receipt_dir"]) if ident["receipt_dir"] else None)
    if out is None or not args.inputs or not args.prereg_url:
        ap.error("--inputs, --prereg-url and --out (or RL_RECEIPT_DIR) are required")
    pkg = k3.Package(out)
    pkg.log("start", fingerprint=fp, seed=args.seed, n_boot=args.n_boot, unit=ident["unit_id"])
    lock = json.loads(LOCK.read_text())
    for name, meta in lock["frozen_files"].items():
        pkg.add_frozen_input(f"receipts/inputs/{name}", args.inputs / name, meta["sha256"])
    pkg.add_receipt("receipts/inputs.lock.json", LOCK.read_bytes())
    res = run(args.inputs, args.seed, args.n_boot)
    rows = [{"kind": "primary_lcb", **res["primary_lcb"]}]
    for cat, pairs in res["secondary_livebench"].items():
        rows += [{"kind": "secondary_livebench", "category": cat, **r} for r in pairs]
    rows += [{"kind": "exploratory_lcb_release_gap", **r} for r in res["exploratory_lcb_release_gap"]]
    rows.append({"kind": "coverage", "livebench_judgments_total": res["livebench_judgments_total"],
                 "livebench_judgments_unmatched": res["livebench_judgments_unmatched"]})
    pkg.write_jsonl("receipts/results.jsonl", json.loads(json.dumps(rows, default=float),
                                                         parse_constant=lambda c: None))
    pkg.log("done", classification=res["primary_lcb"]["classification"])
    env = {"hardware": "cpu (Idleworks CPU executor)", "engine": f"python {k3.host_summary()['python']} numpy",
           "fingerprint": fp, "host": k3.host_summary()}
    pkg.finish(identity=ident, title=TITLE, lane="L3", prereg_path=args.prereg, prereg_url=args.prereg_url,
               arm=args.arm, seed=args.seed, environment=env, verdict=verdict_text(res, args.prereg_url),
               tiers=["P1", "P3"])
    print(json.dumps({"ok": True, "classification": res["primary_lcb"]["classification"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
