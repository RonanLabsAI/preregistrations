"""Receipt-only implementation of the preregistered primary summaries; no model calls."""

import argparse
import json
import random
import statistics
from pathlib import Path

from runner import wilson


def analyze(root):
    rows = [json.loads(x) for x in (root / "receipts/results.jsonl").read_text().splitlines()]
    valid = [r for r in rows if r.get("valid") and r.get("phase") == "measurement"]
    manifest = json.loads((root / "manifest.json").read_text())
    product = manifest["product_id"]
    output = {
        "product": product,
        "run_id": manifest["run_id"],
        "valid_rows": len(valid),
        "invalid_rows": sum(not r.get("valid") for r in rows),
    }
    if not valid:
        output["disposition"] = "no scored rows; smoke and invalid rows excluded"
        return output
    if "needle" in product:
        cells = {}
        for context, depth in sorted({(r["target_tokens"], r["depth"]) for r in valid}):
            sub = [r for r in valid if r["target_tokens"] == context and r["depth"] == depth]
            k = sum(r["correct"] for r in sub)
            n = len(sub)
            cells[f"{context}-{depth}"] = {
                "correct": k,
                "n": n,
                "wilson95": wilson(k, n),
                "min_prompt_eval_count": min(r["prompt_eval_count"] for r in sub),
                "max_prompt_eval_count": max(r["prompt_eval_count"] for r in sub),
            }
        output["cells"] = cells
        contexts = {
            c: [r for r in valid if r["target_tokens"] == c]
            for c in sorted({r["target_tokens"] for r in valid})
        }
        output["all_six_recovered_contexts"] = [
            c for c, sub in contexts.items() if len(sub) == 6 and all(r["correct"] for r in sub)
        ]
    elif "prefix" in product:
        values = {
            cache: {
                r["rep"]: r["batch_wall_s"]
                for r in valid
                if r["cache"] == cache and r["cache_phase"] == "warm" and r["id"].endswith("-0")
            }
            for cache in [False, True]
        }
        reps = sorted(set(values[False]) & set(values[True]))
        rng = random.Random(42)
        if reps:
            draws = []
            for _ in range(2000):
                sample = rng.choices(reps, k=len(reps))
                draws.append(
                    statistics.mean(values[True][i] for i in sample)
                    / statistics.mean(values[False][i] for i in sample)
                )
            draws.sort()
            output["warm_cache_on_off_ratio"] = {
                "point": statistics.mean(values[True].values())
                / statistics.mean(values[False].values()),
                "ci95": [draws[49], draws[1949]],
                "n_paired_batches": len(reps),
            }
        output["output_blue"] = {
            "correct": sum((r.get("answer") or "").strip().lower() == "blue" for r in valid),
            "n": len(valid),
        }
    else:
        output["groups"] = json.loads((root / "receipts/summary.json").read_text())
        if "bakeoff" in product:
            output["strata"] = {}
            for model in sorted({r["model"] for r in valid}):
                for dataset in ["gsm8k", "bbh"]:
                    sub = [
                        r
                        for r in valid
                        if r["model"] == model
                        and str(r["input_id"]).startswith("bbh-") == (dataset == "bbh")
                    ]
                    n = len(sub)
                    k = sum(r["correct"] for r in sub)
                    output["strata"][model + "-" + dataset] = {
                        "correct": k,
                        "n": n,
                        "wilson95": wilson(k, n),
                        "tool_calls": sum(max(0, len(r["trace"]) - 1) for r in sub),
                    }
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("package", type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.package), indent=2))
