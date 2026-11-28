#!/usr/bin/env python3
"""S0 secondary scoring: per-item correctness for one unit, using the public mom-lab scorers.

Usage:
  python3 score.py --prompts prompts.jsonl --results results.jsonl --unit NAME \
      --mom-lab /path/to/RonanLabsAI-mom-lab-checkout --python /path/to/venv/python --out items.json

The mom-lab checkout must be at the commit named in the protocol. This script lays the
frozen prompts and the unit's outputs out in harness.generate's directory layout in a
temporary directory, runs `python -m harness.aggregate` there, and writes
{"<id>": true|false} (correct = score 1.0) plus per-benchmark scores. Scoring is secondary:
it is used only to report how far scores move when outputs are not byte-identical.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--mom-lab", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    outs = {}
    for l in open(args.results, encoding="utf-8"):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except ValueError:
            continue  # a line cut short by an interruption; analyze.py counts these
        if isinstance(r.get("output"), str):
            outs.setdefault(r["id"], r)  # first record wins, as in analyze.py

    with tempfile.TemporaryDirectory() as tmp:
        data, res = Path(tmp) / "data", Path(tmp) / "results"
        for p in prompts:
            ev = data / "eval" / p["domain"] / p["bench"] / "dev.jsonl"
            gen = res / "s0" / args.unit / "gen" / p["domain"] / f"{p['bench']}.dev.jsonl"
            ev.parent.mkdir(parents=True, exist_ok=True)
            gen.parent.mkdir(parents=True, exist_ok=True)
            row = {"id": p["id"], "messages": p["messages"], "reference": p["reference"],
                   "meta": p["meta"], "max_tokens": p["max_tokens"]}
            with open(ev, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if p["id"] in outs:
                o = outs[p["id"]]
                g = {"id": o["id"], "output": o["output"], "finish_reason": o.get("finish_reason"),
                     "n_out_tokens": o.get("n_out_tokens")}
                with open(gen, "a", encoding="utf-8") as f:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
        subprocess.run([args.python, "-m", "harness.aggregate", "--run-id", args.unit, "--step", "s0",
                        "--split", "dev", "--benchmarks", "gsm8k,math500,ifeval",
                        "--data-root", str(data), "--results-root", str(res)],
                       cwd=args.mom_lab, check=True)
        scores = json.load(open(res / "s0" / args.unit / "scores.json", encoding="utf-8"))

    items, per = {}, {}
    for bench, b in scores["benchmarks"].items():
        for i, c in zip(b["ids"], b["correct"]):
            items[i] = float(c) == 1.0
        per[bench] = {"score": b["score"], "n": b["n"], "n_missing": b["n_missing"]}
    Path(args.out).write_text(json.dumps(items, indent=0))
    Path(args.out).with_suffix(".bench.json").write_text(json.dumps(per, indent=1))
    print(json.dumps(per))
    return 0


if __name__ == "__main__":
    sys.exit(main())
