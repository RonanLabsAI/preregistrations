"""Build R1 item sets (card: Models and items).

`items.py build` writes configs/r1/items/{transfer_core,full_dev}.jsonl + SHA256SUMS.

Transfer core (2,244): paper-2 dev GSM8K 659, MATH-500 250, IFEval 270, LiveCodeBench 65
(read exactly as harness.generate does: data/eval/<domain>/<bench>/dev.jsonl, full row kept),
plus a 1,000-item MMLU-Pro subsample of data/eval/general/mmlu_pro/dev.jsonl (1,499 rows),
stratified by meta.category, proportional allocation with largest-remainder rounding,
random.Random(20261128). Subsample is emitted in original file order.

Full set (7,260): see FLAG in configs/r1/pin.md and the report. The only 7,260 composition in
paper 2's history is the protocol set (notebook/2026-10-29.md; results/protocol/p-greedy-4k/
scores.json): GSM8K+MATH-500+IFEval+LiveCodeBench dev plus MMLU-Pro dev_full.jsonl (6,016,
the pre-reduction dev file kept beside dev.jsonl). All-core dev.jsonl files sum to 7,317,
not 7,260, so the card's number cannot be met from dev.jsonl alone.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "configs" / "r1" / "items"
SEED = 20261128
TRANSFER_SPECS = [("math", "gsm8k"), ("math", "math500"), ("general", "ifeval"), ("code", "livecodebench")]
BENCH_OF = {"gsm8k": "gsm8k", "math500": "math500", "ifeval": "ifeval", "livecodebench": "livecodebench",
            "mmlu_pro": "mmlu_pro"}

def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

def eval_path(domain, bench, name="dev"):
    return DATA / "eval" / domain / bench / f"{name}.jsonl"

def stratified_allocation(counts: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation, largest-remainder rounding; ties by category name."""
    n = sum(counts.values())
    if n == 0 or total <= 0:
        return {k: 0 for k in counts}
    quotas = {k: v * total / n for k, v in counts.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    rem = total - sum(alloc.values())
    order = sorted(counts, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in order[:rem]:
        alloc[k] += 1
    return alloc

def sample_mmlu(rows: list[dict], n: int = 1000, seed: int = SEED) -> list[dict]:
    """Stratified subsample by meta.category; selection seeded, output in file order."""
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(str((r.get("meta") or {}).get("category")), []).append(r)
    alloc = stratified_allocation({k: len(v) for k, v in by_cat.items()}, n)
    rng = random.Random(seed)
    chosen = set()
    for cat in sorted(by_cat):
        k = alloc.get(cat, 0)
        if k:
            chosen.update(r["id"] for r in rng.sample(by_cat[cat], k))
    return [r for r in rows if r["id"] in chosen]

def _tag(rows, bench):
    return [dict(r, benchmark=bench) for r in rows]

def build_items(data_root=None) -> dict:
    global DATA
    if data_root is not None:
        DATA = Path(data_root)
    transfer, counts = [], {}
    for domain, bench in TRANSFER_SPECS:
        rows = read_jsonl(eval_path(domain, bench))
        counts[bench] = len(rows)
        transfer += _tag(rows, bench)
    mmlu = read_jsonl(eval_path("general", "mmlu_pro"))
    sub = sample_mmlu(mmlu, 1000)
    counts["mmlu_pro_transfer"] = len(sub)
    counts["mmlu_pro_pool"] = len(mmlu)
    transfer += _tag(sub, "mmlu_pro")
    # Full 7,260: protocol composition (see module docstring).
    full = []
    for domain, bench in TRANSFER_SPECS:
        full += _tag(read_jsonl(eval_path(domain, bench)), bench)
    full += _tag(read_jsonl(eval_path("general", "mmlu_pro", "dev_full")), "mmlu_pro")
    counts["full_dev"] = len(full)
    return {"transfer_core": transfer, "full_dev": full, "counts": counts}

def sha256_line(path, rel: str) -> str:
    return f"{hashlib.sha256(Path(path).read_bytes()).hexdigest()}  {rel}"

def write_items(out_dir=None) -> dict:
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    built = build_items()
    lines = []
    for name in ["transfer_core", "full_dev"]:
        p = out / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in built[name]), encoding="utf-8")
        lines.append(sha256_line(p, f"{name}.jsonl"))
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return built

def main(argv=None):
    ap = argparse.ArgumentParser(prog="items.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        built = write_items()
        print(json.dumps(built["counts"], indent=2, sort_keys=True))
        print(f"transfer_core={len(built['transfer_core'])} full_dev={len(built['full_dev'])}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
