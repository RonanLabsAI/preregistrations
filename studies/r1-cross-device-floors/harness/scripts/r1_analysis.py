"""R1 metrics and analysis (card: Metrics and analysis). Pure python + numpy; no vllm.

Per item/run pair: graded flip (primary), extracted-answer flip, text flip.
Floors per condition; equal-benchmark-weight means; one shared cluster-bootstrap matrix
(2,000 resamples, clusters = MMLU-Pro category else benchmark, rng 0); exact McNemar;
churn worse/better/wrong-to-wrong; set-size SE curve; rescore check; H1a/H1b/H2/H5/H6 rules.
Run identities come from run_meta.json files + configs/r1/runs.json manifest.
CLI writes --out (default results/analysis/r1_main.json; not written by tests).
"""
from __future__ import annotations
import argparse, json, math, sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def extract_for(benchmark: str, text: str):
    """Harness scorers' answer extraction, dispatched per benchmark."""
    if text is None:
        return None
    try:
        if benchmark == "mmlu_pro":
            from harness.score.general import extract_letter
            return extract_letter(text)
        if benchmark in ("gsm8k", "math500"):
            from harness.score.math_ import extract_answer
            return extract_answer(text)[0]
        if benchmark in ("livecodebench", "humaneval_plus", "mbpp_plus"):
            from harness.score.code import extract_code
            return extract_code(text)
    except Exception:
        return text
    return text  # ifeval / others: extracted answer is the (stripped) text itself

def _by_id(recs):
    return {r["id"]: r for r in recs}

def _extracted(r):
    if r.get("extracted") is not None:
        return r["extracted"]
    return extract_for(r.get("benchmark", ""), r.get("text") or r.get("output") or "")

def _text(r):
    return r.get("text") if r.get("text") is not None else r.get("output")

def pair_flips(a, b) -> dict:
    """Graded/extracted/text flip rates (equal-benchmark-weight) + churn for one run pair."""
    A, B = _by_id(a), _by_id(b)
    ids = [i for i in A if i in B]
    per_bench: dict[str, list] = {}
    worse = better = wtw = 0
    net_per_bench: dict[str, list] = {}
    for i in ids:
        ra, rb = A[i], B[i]
        bench = ra.get("benchmark") or rb.get("benchmark")
        graded = float(ra["correct"] != rb["correct"])
        text_f = float(_text(ra) != _text(rb))
        ext_f = float(_extracted(ra) != _extracted(rb))
        per_bench.setdefault(bench, []).append((graded, text_f, ext_f))
        net_per_bench.setdefault(bench, []).append(float(rb["correct"]) - float(ra["correct"]))
        if ra["correct"] > rb["correct"]:
            worse += 1
        elif ra["correct"] < rb["correct"]:
            better += 1
        elif ra["correct"] == 0 and rb["correct"] == 0 and _extracted(ra) != _extracted(rb):
            wtw += 1
    def eq_mean(idx):
        return float(np.mean([np.mean([x[idx] for x in v]) for v in per_bench.values()])) if per_bench else 0.0
    return {"n": len(ids), "graded_flip": eq_mean(0), "extracted_flip": eq_mean(2), "text_flip": eq_mean(1),
            "per_benchmark": {k: {"graded_flip": float(np.mean([x[0] for x in v])),
                                  "text_flip": float(np.mean([x[1] for x in v])), "n": len(v)}
                              for k, v in per_bench.items()},
            "churn": {"worse": worse, "better": better, "wrong_to_wrong": wtw,
                      "net_accuracy_change": float(np.mean([np.mean(v) for v in net_per_bench.values()])) if net_per_bench else 0.0},
            "mcnemar": {"worse": worse, "better": better, "p_exact": exact_mcnemar(worse, better)}}

CONFIRMATORY_MODEL = "qwen3-4b-instruct-2507"
CONFIRMATORY_DEVICES = ["gb300", "rtx6000", "gb10", "rtx5080"]
H1A_DEVICES = ("gb300", "rtx6000")
EXPECTED_IMAGE_DIGEST = "vllm/vllm-openai@sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd"
TRANSFER_CORE_ITEMS = "configs/r1/items/transfer_core.jsonl"
CROSS_STRATUM_DEVICE = "rtx5080"  # its other-digest H1b pairs are device+build pairs, reported separately


def _cluster_for(rec) -> str:
    """Bootstrap cluster: MMLU-Pro category; every other item is its own cluster."""
    if (rec or {}).get("benchmark") == "mmlu_pro":
        return (rec.get("cluster") or rec.get("benchmark") or rec.get("id"))
    return (rec or {}).get("id")

def byte_identity(a, b) -> dict:
    A, B = _by_id(a), _by_id(b)
    ids = [i for i in A if i in B]
    mismatch = []
    for i in list(A) + [i for i in B if i not in A]:
        if i not in A or i not in B:
            mismatch.append(i); continue
        ra, rb = A[i], B[i]
        if _text(ra) != _text(rb):
            mismatch.append(i); continue
        if ("token_ids" in ra or "token_ids" in rb) and list(ra.get("token_ids") or []) != list(rb.get("token_ids") or []):
            mismatch.append(i)
    return {"passed": len(mismatch) == 0 and len(ids) == len(A) == len(B),
            "n": len(ids), "n_mismatch": len(mismatch), "mismatch_ids": mismatch}

def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, 201):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + aa / c if abs(1.0 + aa / c) >= 1e-300 else 1e-300
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < 3e-7:
            break
    return h

def _reg_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b

def clopper_pearson_upper(k, n, confidence=0.95) -> float:
    """One-sided Clopper-Pearson upper bound for k successes in n trials."""
    k, n = int(k), int(n)
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if k < 0:
        raise ValueError("k must be >= 0")
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _reg_beta(mid, k + 1, n - k) < confidence:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)

def zero_flip_bounds(pair_vectors, clusters=None, benches=None, reps=2000, seed=0, shared_indices=None) -> dict:
    """Cluster-bootstrap 95th-percentile upper bound (shared resamples) next to Clopper-Pearson.

    `pair_vectors` is a list of per-item graded-flip arrays (one per run pair, aligned).
    The bootstrap statistic is the equal-benchmark-weight mean over the shared resamples.
    """
    vecs = [np.asarray(v, dtype=float) for v in pair_vectors if len(v)]
    if not vecs:
        return {"k": 0, "n": 0, "rate": 0.0, "cluster_bootstrap_upper95": 0.0,
                "clopper_pearson_upper95": 1.0, "n_shared_resamples": 0}
    mean_vec = np.mean(vecs, axis=0)
    n_items = len(mean_vec)
    benches = list(benches) if benches is not None else ["all"] * n_items
    clusters = list(clusters) if clusters is not None else [str(i) for i in range(n_items)]
    idx = shared_indices if shared_indices is not None else shared_cluster_indices(
        clusters, reps=reps, seed=seed, strata=benches)
    bench_arr = np.array(benches)
    boot = []
    for row in idx:
        b_res = bench_arr[row]
        boot.append(float(np.mean([mean_vec[row][b_res == b].mean() for b in sorted(set(b_res))])) if len(row) else 0.0)
    k = int(round(float(sum(v.sum() for v in vecs))))
    n = int(sum(len(v) for v in vecs))
    return {"k": k, "n": n, "rate": _eq_bench_mean(mean_vec, benches),
            "cluster_bootstrap_upper95": float(np.percentile(boot, 95)) if boot else 0.0,
            "clopper_pearson_upper95": clopper_pearson_upper(k, n),
            "n_shared_resamples": len(idx)}

def exact_mcnemar(worse: int, better: int) -> float:
    """Exact two-sided McNemar: binomial test on discordant pairs, p=0.5."""
    n = worse + better
    if n == 0:
        return 1.0
    k = min(worse, better)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))

def set_size_curve(discordant, n_total, p, ns):
    """Card formulas: paired SE sqrt(D)/n; single-score SE sqrt(p(1-p)/n). Diagnostic only."""
    return [{"n": int(n), "paired_se_formula": float(np.sqrt(discordant) / n),
             "single_se_formula": float(np.sqrt(p * (1 - p) / n)),
             "note": "D=discordant items in the full sample; formula as stated in card"} for n in ns]

def rescore_check(rows, scorer) -> dict:
    """Score every stored output twice; report differing rows (card: rescore check)."""
    differing = []
    for r in rows:
        if scorer(r) != scorer(r):
            differing.append(r["id"])
    return {"n": len(rows), "n_differing": len(differing), "differing_ids": differing}

def shared_cluster_indices(clusters, reps=2000, seed=0, strata=None):
    """One shared list of bootstrap resamples (card: cluster bootstrap, clusters = MMLU-Pro category, else the item;
    Rane 2611.29663 clusters MMLU by subject). Within each stratum (benchmark), whole clusters are drawn with
    replacement and all their items taken, so resample sizes vary when clusters differ in size; every paired
    statistic uses the same resamples. `clusters[i]` is item i's cluster label; `strata[i]` its benchmark
    (default: one stratum)."""
    clusters = list(clusters)
    strata = list(strata) if strata is not None else ["all"] * len(clusters)
    rng = np.random.default_rng(seed)
    plan = []
    for s in sorted(set(strata)):
        labels = sorted({c for c, st in zip(clusters, strata) if st == s})
        members = {c: np.array([i for i, (cc, st) in enumerate(zip(clusters, strata)) if st == s and cc == c])
                   for c in labels}
        plan.append((labels, members))
    out = []
    for _ in range(reps):
        parts = []
        for labels, members in plan:
            pick = rng.integers(0, len(labels), size=len(labels))
            parts.extend(members[labels[k]] for k in pick)
        out.append(np.concatenate(parts))
    return out

def _per_item_flip_vectors(run_lists_a, run_lists_b, ids):
    """Per-item graded-flip probability averaged over all A x B pairs (aligned to ids)."""
    def mat(runs):
        return np.array([[float(r["correct"]) for r in sorted(run, key=lambda x: ids.index(x["id"]))] if False else
                         [float(_by_id(run)[i]["correct"]) for i in ids] for run in runs])
    ma, mb = mat(run_lists_a), mat(run_lists_b)
    flips = []
    for x in ma:
        for y in mb:
            flips.append((x != y).astype(float))
    return np.mean(flips, axis=0) if flips else np.zeros(len(ids))

def _eq_bench_mean(values, benches):
    benches = np.array(benches)
    return float(np.mean([values[benches == b].mean() for b in sorted(set(benches))])) if len(values) else 0.0

def h1a(runs: dict, runs_a_key="gb300", runs_b_key="rtx6000", reps=2000, seed=0) -> dict:
    """Cross-device graded flip minus mean of the two warm-warm floors, shared bootstrap.

    `runs` maps run_id -> records; keys starting with runs_a_key/runs_b_key select each device.
    Clusters = record cluster (MMLU-Pro category else benchmark); rng 0 (numpy default_rng(0)).
    Ratio reported only if the floor's upper CI >= 0.5% (card H1a), else None.
    """
    a_runs = [v for k, v in runs.items() if k.startswith(runs_a_key)]
    b_runs = [v for k, v in runs.items() if k.startswith(runs_b_key)]
    if not a_runs or not b_runs:
        raise ValueError("need runs for both devices")
    ids = [r["id"] for r in a_runs[0]]
    benches = [_by_id(a_runs[0])[i].get("benchmark") for i in ids]
    clusters = [(_by_id(a_runs[0])[i].get("cluster") if _by_id(a_runs[0])[i].get("benchmark") == "mmlu_pro" else None)
                or i for i in ids]  # MMLU-Pro: category clusters; other benchmarks: each item its own cluster
    cross_v = _per_item_flip_vectors(a_runs, b_runs, ids)
    floor_a = _per_item_flip_vectors(a_runs, a_runs, ids) if len(a_runs) > 1 else np.zeros(len(ids))
    # within-device: exclude self-pairs (a run vs itself is identically zero, would dilute)
    def within(runs_):
        vecs = [_per_item_flip_vectors([x], [y], ids) for x, y in combinations(runs_, 2)]
        return np.mean(vecs, axis=0) if vecs else np.zeros(len(ids))
    floor_a, floor_b = within(a_runs), within(b_runs)
    cross_rate = _eq_bench_mean(cross_v, benches)
    fa, fb = _eq_bench_mean(floor_a, benches), _eq_bench_mean(floor_b, benches)
    floor_mean = (fa + fb) / 2.0
    idx = shared_cluster_indices(clusters, reps=reps, seed=seed, strata=benches)
    bench_arr = np.array(benches)
    boot_excess, boot_floor = [], []
    for row in idx:
        b_res = bench_arr[row]
        c = float(np.mean([cross_v[row][b_res == b].mean() for b in sorted(set(b_res))]))
        f = (float(np.mean([floor_a[row][b_res == b].mean() for b in sorted(set(b_res))])) +
             float(np.mean([floor_b[row][b_res == b].mean() for b in sorted(set(b_res))]))) / 2.0
        boot_excess.append(c - f); boot_floor.append(f)
    ci = [float(np.percentile(boot_excess, 2.5)), float(np.percentile(boot_excess, 97.5))]
    floor_ci = [float(np.percentile(boot_floor, 2.5)), float(np.percentile(boot_floor, 97.5))]
    ratio = (cross_rate / floor_mean) if (floor_ci[1] >= 0.005 and floor_mean > 0) else None
    return {"cross_rate": cross_rate, "floor_a": fa, "floor_b": fb, "floor_mean": floor_mean,
            "excess": cross_rate - floor_mean, "ci95": ci, "floor_ci95": floor_ci,
            "supported": bool(ci[0] > 0), "ratio": ratio,
            "note": "intervals are item-sampling error conditional on the runs (card)"}

def interval_rule(diff, ci) -> str:
    """H2-style rule: supported only if interval excludes 0."""
    return "supported" if (ci[0] > 0 or ci[1] < 0) else "inconclusive"

def h6_label(loss_fraction_a, loss_fraction_b) -> str:
    return "closer to Rane than to BlockLock" if (loss_fraction_a < 0.5 and loss_fraction_b < 0.5) else "not closer to Rane than to BlockLock"

def floors_for_runs(named_runs: dict[str, list]) -> dict:
    """All within-set pairs: per-condition floors from pair_flips (named_runs grouped externally)."""
    out = {}
    for name, runs in named_runs.items():
        pairs, vecs, base_ids, base = [], [], [], []
        for x, y in combinations(runs, 2):
            pairs.append(pair_flips(x, y))
            vec, ids = _pair_flip_vector(x, y)
            if vecs == [] and len(vec):
                base_ids, base = ids, x
            vecs.append(vec)
        if pairs:
            by_id = _by_id(base)
            bounds = zero_flip_bounds(vecs,
                                      clusters=[(_cluster_for(by_id[i]) if i in by_id else i) for i in base_ids],
                                      benches=[(by_id[i].get("benchmark") if i in by_id else None) for i in base_ids]) if vecs else {}
            out[name] = {"n_pairs": len(pairs),
                         "graded_flip_mean": float(np.mean([p["graded_flip"] for p in pairs])),
                         "graded_flip_max_pair": float(np.max([p["graded_flip"] for p in pairs])),
                         "clopper_pearson_upper95": bounds.get("clopper_pearson_upper95"),
                         "cluster_bootstrap_upper95": bounds.get("cluster_bootstrap_upper95"),
                         "zero_flip_bounds": bounds,
                         "pairs": pairs}
    return out

# ---- manifest-driven CLI -------------------------------------------------
def split_passes(records) -> dict:
    """Group records by their `pass` label (pass1/pass2), preserving order."""
    out: dict[str, list] = {}
    for r in records:
        out.setdefault(r.get("pass") or "pass1", []).append(r)
    return {k: out[k] for k in sorted(out)}

def load_run(run_dir: Path, items_by_id: dict) -> list:
    """Load one run dir. Both same-process passes are kept (records carry `pass`).

    Any generated item with no score in scores.json aborts (ValueError listing the
    ids); a missing score is never defaulted to 0.
    """
    scores = {}
    sp = Path(run_dir) / "scores.json"
    if sp.exists():
        data = json.loads(sp.read_text())
        for bench, e in (data.get("benchmarks") or {}).items():
            for i, c in zip(e.get("ids", []), e.get("correct", [])):
                scores[i] = (bench, float(c))
    recs, missing = [], []
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("pass*/gen/**/*.jsonl")) + sorted(run_dir.glob("gen/**/*.jsonl"))
    for gp in files:
        rel = gp.relative_to(run_dir).parts
        pass_label = rel[0] if rel and rel[0].startswith("pass") else "pass1"
        for line in gp.read_text().splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            if g["id"] not in scores:
                if g["id"] not in missing:
                    missing.append(g["id"])
                continue
            bench, correct = scores[g["id"]]
            bench = bench or g.get("benchmark")
            meta = items_by_id.get(g["id"], {})
            # MMLU-Pro: category cluster; non-MMLU: each item its own cluster (Deviation 4 #7)
            cluster = ((meta.get("meta") or {}).get("category") or bench) if bench == "mmlu_pro" else g["id"]
            recs.append({"id": g["id"], "benchmark": bench, "cluster": cluster, "correct": correct,
                         "text": g.get("text", g.get("output", "")), "token_ids": g.get("token_ids"),
                         "pass": pass_label})
    if missing:
        raise ValueError(f"unscored items (missing in scores.json), never defaulted to 0: {missing}")
    return recs

def _primary_records(run_entry: dict) -> list:
    passes = run_entry.get("passes") or split_passes(run_entry["records"])
    if "pass1" in passes:
        return passes["pass1"]
    return next(iter(passes.values()), [])

def _prompt_total(run_entry: dict):
    stats = (run_entry.get("run_meta") or {}).get("scheduler_stats") or {}
    return stats.get("prompt_tokens")

def _is_h1a(m: dict) -> bool:
    return (m.get("arm") == "transfer" and m.get("kind") == "restart" and m.get("cache") == "warm"
            and m.get("model") == CONFIRMATORY_MODEL and m.get("mode") == "default")

def _is_h1b(m: dict) -> bool:
    return (m.get("arm") == "transfer" and m.get("kind") == "restart"
            and m.get("model") == CONFIRMATORY_MODEL and m.get("mode") == "bi")

def decide_strata(metas_by_device: dict) -> dict:
    """Stratum from run_meta image digest (+ arch recorded per device).

    Same digest as the confirmatory pair's common digest -> S; anything else is
    not-S (reported, never assumed). Deviation 1: digest, not host arch, decides.
    """
    digests = {d: (m or {}).get("image_digest") for d, m in metas_by_device.items()}
    ref = digests.get("gb300")
    same_pair = bool(ref) and digests.get("rtx6000") == ref
    devices = {d: {"image_digest": digests.get(d), "arch": (m or {}).get("arch")
                   or (((m or {}).get("env_fingerprint") or {}).get("arch")),
                   "stratum": "S" if (same_pair and digests.get(d) == ref) else "not-S"}
               for d, m in metas_by_device.items()}
    return {"devices": devices, "confirmatory_pair_same_digest": same_pair,
            "reference_digest": ref if same_pair else None}

def _assert_scored_ids(named_records: dict, expected_ids, context: str):
    """Abort unless every included run's scored ids equal the transfer-core id set."""
    if not expected_ids:
        return
    for rid, recs in named_records.items():
        got = {r["id"] for r in recs}
        if got != set(expected_ids):
            raise ValueError(f"{context} scored ids must equal exactly the transfer-core id set "
                             f"({len(expected_ids)} ids); run {rid} has {len(got)} "
                             f"(missing {sorted(set(expected_ids) - got)[:5]}, extra {sorted(got - set(expected_ids))[:5]})")


def _h1b(device_runs: dict, prompt_by_run: dict, smoke_ids=None, required_devices=None,
         missing_runs=None, cross_stratum_pairs=None) -> dict:
    required = list(required_devices) if required_devices is not None else list(CONFIRMATORY_DEVICES)
    present = [d for d in CONFIRMATORY_DEVICES if device_runs.get(d)]
    missing = [d for d in required if not device_runs.get(d)]
    within, cross = [], []
    for dev in present:
        for (r1, rec1), (r2, rec2) in combinations(sorted(device_runs[dev]), 2):
            b = byte_identity(rec1, rec2)
            within.append({"device": dev, "runs": [r1, r2], "passed": b["passed"],
                           "mismatch_ids": b["mismatch_ids"], "n": b["n"], "n_mismatch": b["n_mismatch"]})
    for d1, d2 in combinations(present, 2):
        run_pairs, mismatch = [], []
        for r1, rec1 in sorted(device_runs[d1]):
            for r2, rec2 in sorted(device_runs[d2]):
                b = byte_identity(rec1, rec2)
                run_pairs.append({"runs": [r1, r2], "passed": b["passed"],
                                  "mismatch_ids": b["mismatch_ids"], "n": b["n"]})
                for i in b["mismatch_ids"]:
                    if i not in mismatch:
                        mismatch.append(i)
        def _tot(dev):
            vals = {prompt_by_run.get(r) for r, _ in device_runs[dev]}
            vals.discard(None)
            return vals.pop() if len(vals) == 1 else sorted(vals, key=str)
        t1, t2 = _tot(d1), _tot(d2)
        cross.append({"devices": [d1, d2], "passed": all(p["passed"] for p in run_pairs),
                      "mismatch_ids": mismatch, "run_pairs": run_pairs,
                      "prompt_token_totals": {d1: t1, d2: t2},
                      "prompt_tokens_equal": bool(t1 is not None and t1 == t2)})
    emitted = within + cross
    incomplete = bool(missing or missing_runs)
    passed = bool(not incomplete and emitted and all(p["passed"] for p in emitted))
    status = "incomplete" if incomplete else ("passed" if passed else "failed")
    sep = list(cross_stratum_pairs or [])
    return {"within_device_pairs": within, "cross_device_pairs": cross,
            "devices_present": present, "missing_devices": missing,
            "missing_runs": missing_runs or {}, "status": status, "passed": passed,
            "cross_stratum_pairs": sep, "device_build_pairs": sep,
            "smoke_item_ids_excluded_from_citation": list(smoke_ids or []),
            "note": "smoke items stay in the item set; they are only excluded from any H1b citation note"}

def _pair_flip_vector(a, b):
    A, B = _by_id(a), _by_id(b)
    ids = [r["id"] for r in a if r["id"] in B]
    vec = np.array([float(A[i]["correct"] != B[i]["correct"]) for i in ids]) if ids else np.zeros(0)
    return vec, ids

def analyze_manifest(manifest: dict, items_by_id: dict, results_root: Path, smoke_ids=None) -> dict:
    runs = {}
    skipped_aborted, skipped_warmup = [], []
    expected_ids = set(items_by_id or {})
    for entry in manifest.get("runs", []):
        if entry.get("arm") == "warmup" or entry.get("kind") == "warmup":
            skipped_warmup.append(entry["run_id"]); continue  # uncounted; never in floors/H1
        rd = Path(entry.get("run_dir") or (Path(results_root) / entry.get("step", "r1") / entry["run_id"]))
        if rd.exists():
            run_meta = {}
            mp = rd / "run_meta.json"
            if mp.exists():
                try:
                    run_meta = json.loads(mp.read_text())
                except Exception:
                    run_meta = {}
            if "aborted" in (run_meta or {}):
                skipped_aborted.append(entry["run_id"]); continue  # only clean metas are data
            records = load_run(rd, items_by_id)  # ValueError on unscored items aborts the analysis
            runs[entry["run_id"]] = {"meta": entry, "records": records,
                                     "passes": split_passes(records), "run_meta": run_meta}
    result = {"n_runs_loaded": len(runs), "runs": sorted(runs), "floors": {}, "byte_identity": {},
              "skipped_aborted_runs": skipped_aborted, "skipped_warmup_runs": skipped_warmup,
              "cross_stratum_runs": [{"run_id": rid, "device": r["meta"].get("device"),
                                      "image_digest": (r.get("run_meta") or {}).get("image_digest")}
                                     for rid, r in sorted(runs.items())
                                     if (r.get("run_meta") or {}).get("image_digest") != EXPECTED_IMAGE_DIGEST]}
    # strata decided from run_meta image digest + arch (Deviation 1 / C5)
    metas_by_device: dict[str, dict] = {}
    for r in runs.values():
        metas_by_device.setdefault(r["meta"].get("device"), r.get("run_meta") or {})
    result["strata"] = decide_strata(metas_by_device)
    result["prompt_token_totals_by_run"] = {rid: _prompt_total(r) for rid, r in runs.items()}
    result["preemptions_by_run"] = {rid: ((r.get("run_meta") or {}).get("scheduler_stats") or {}).get("n_preemptions")
                                    for rid, r in runs.items()}
    # Floors per kind, never pooled: restart cache-pairs, same-process pass1-vs-pass2, order vs warm restarts
    from collections import defaultdict
    groups = defaultdict(list)
    for rid, r in runs.items():
        # floor key includes arm + items file: matched_rane/full_set never enter transfer floors
        groups[(r["meta"].get("device"), r["meta"].get("model"), r["meta"].get("mode"),
                r["meta"].get("arm"), r["meta"].get("items") or TRANSFER_CORE_ITEMS)].append((rid, r))
    for key, lst in groups.items():
        cond_pairs: dict[str, list] = defaultdict(list)   # cond -> [(pair_flips, vector, ids, base_recs)]
        def _add(cond, ra, rb):
            vec, ids = _pair_flip_vector(ra, rb)
            cond_pairs[cond].append((pair_flips(ra, rb), vec, ids, ra))
        restarts = [(rid, r) for rid, r in lst if r["meta"].get("kind") == "restart"]
        for (rid1, r1), (rid2, r2) in combinations(restarts, 2):
            c1, c2 = r1["meta"].get("cache") or "warm", r2["meta"].get("cache") or "warm"
            cond = "-".join(sorted([c1, c2], key=lambda c: {"cold": 0, "warm": 1}.get(c, 1)))
            if cond == "warm-cold":
                cond = "cold-warm"
            _add(cond, _primary_records(r1), _primary_records(r2))
        for rid, r in lst:
            if r["meta"].get("kind") == "same_process":
                passes = r.get("passes") or {}
                if "pass1" in passes and "pass2" in passes:
                    _add("same-process", passes["pass1"], passes["pass2"])
        warm_restarts = [(rid, r) for rid, r in restarts if (r["meta"].get("cache") or "warm") == "warm"]
        for rid, r in lst:
            if r["meta"].get("kind") == "order":
                for wrid, wr in warm_restarts:
                    _add("order", _primary_records(r), _primary_records(wr))
        cond_out = {}
        for cond, ps in cond_pairs.items():
            vecs = [v for _, v, _, _ in ps if len(v)]
            base = ps[0][3] if ps else []
            by_id = _by_id(base)
            ids = ps[0][2] if ps else []
            bounds = zero_flip_bounds(vecs,
                                      clusters=[(_cluster_for(by_id[i]) if i in by_id else i) for i in ids],
                                      benches=[(by_id[i].get("benchmark") if i in by_id else None) for i in ids],
                                      reps=2000, seed=0) if vecs else {}
            cond_out[cond] = {"graded_flip_mean": float(np.mean([p["graded_flip"] for p, *_ in ps])),
                              "graded_flip_max_pair": float(np.max([p["graded_flip"] for p, *_ in ps])),
                              "n_pairs": len(ps), "pairs": [p for p, *_ in ps],
                              "clopper_pearson_upper95": bounds.get("clopper_pearson_upper95"),
                              "cluster_bootstrap_upper95": bounds.get("cluster_bootstrap_upper95"),
                              "zero_flip_bounds": bounds}
        result["floors"]["|".join(str(x) for x in key)] = cond_out
    # Legacy byte_identity map (all bi run pairs) kept for descriptive use; H1b below is the confirmatory verdict
    for (r1id, r1), (r2id, r2) in combinations(runs.items(), 2):
        if r1["meta"].get("mode") == "bi" and r2["meta"].get("mode") == "bi":
            result["byte_identity"][f"{r1id} vs {r2id}"] = byte_identity(_primary_records(r1), _primary_records(r2))
    # H1a: only arm=transfer kind=restart cache=warm model=Q4B mode=default; exactly 3 per confirmatory device
    h1a_by_device: dict[str, list] = defaultdict(list)
    for rid, r in runs.items():
        if _is_h1a(r["meta"]):
            h1a_by_device[r["meta"].get("device")].append((rid, r))
    if h1a_by_device.get(H1A_DEVICES[0]) and h1a_by_device.get(H1A_DEVICES[1]):
        counts = {d: len(h1a_by_device.get(d, [])) for d in H1A_DEVICES}
        if any(c != 3 for c in counts.values()):
            raise ValueError(f"H1a requires exactly 3 clean warm transfer restarts per confirmatory device "
                             f"(arm=transfer, kind=restart, cache=warm, model={CONFIRMATORY_MODEL}, mode=default); "
                             f"got {counts}")
        for dev in H1A_DEVICES:
            for rid, r in h1a_by_device[dev]:
                dg = (r.get("run_meta") or {}).get("image_digest")
                if dg != EXPECTED_IMAGE_DIGEST:
                    raise ValueError(f"H1a confirmatory run {rid} image_digest {dg!r} != expected digest "
                                     f"{EXPECTED_IMAGE_DIGEST!r}; aborting analysis")
        _assert_scored_ids({rid: _primary_records(r) for dev in H1A_DEVICES
                            for rid, r in h1a_by_device[dev]}, expected_ids, "H1a")
        keyed = {f"{dev}-{rid}": _primary_records(r)
                 for dev in H1A_DEVICES for rid, r in sorted(h1a_by_device[dev])}
        h = h1a(keyed, H1A_DEVICES[0], H1A_DEVICES[1])
        totals_by_run = {rid: _prompt_total(r) for dev in H1A_DEVICES for rid, r in sorted(h1a_by_device[dev])}
        if any(v is None for v in totals_by_run.values()):
            raise ValueError(f"H1a prompt-token totals missing (exact metric vllm:prompt_tokens) "
                             f"for runs: {[k for k, v in totals_by_run.items() if v is None]}")
        if len(set(totals_by_run.values())) != 1:
            raise ValueError(f"H1a prompt-token totals differ across the six runs "
                             f"(exact metric vllm:prompt_tokens): {totals_by_run}")
        totals = {dev: totals_by_run[sorted(h1a_by_device[dev])[0][0]] for dev in H1A_DEVICES}
        h["prompt_token_totals"] = totals
        h["prompt_token_totals_by_run"] = totals_by_run
        h["prompt_tokens_equal"] = True
        h["run_ids"] = {dev: sorted(rid for rid, _ in h1a_by_device[dev]) for dev in H1A_DEVICES}
        result["h1a"] = h
    else:
        result["h1a"] = {"missing_devices": [d for d in H1A_DEVICES if not h1a_by_device.get(d)],
                         "n_runs_by_device": {d: len(h1a_by_device.get(d, [])) for d in H1A_DEVICES}}
    # H1b: only clean arm=transfer kind=restart (cold+warm) mode=bi model=Q4B, digest-checked
    h1b_by_device: dict[str, list] = defaultdict(list)
    h1b_cache_by_device: dict[str, list] = defaultdict(list)
    separate_runs: list = []  # (rid, records, device, digest) for the cross-stratum device
    prompt_by_run = {}
    for rid, r in sorted(runs.items()):
        if not _is_h1b(r["meta"]):
            continue
        dev = r["meta"].get("device")
        dg = (r.get("run_meta") or {}).get("image_digest")
        if dg != EXPECTED_IMAGE_DIGEST:
            if dev == CROSS_STRATUM_DEVICE:
                separate_runs.append((rid, _primary_records(r), dev, dg)); continue
            raise ValueError(f"H1b confirmatory run {rid} image_digest {dg!r} != expected digest "
                             f"{EXPECTED_IMAGE_DIGEST!r}; aborting analysis")
        h1b_by_device[dev].append((rid, _primary_records(r)))
        h1b_cache_by_device[dev].append((rid, r["meta"].get("cache") or "warm"))
        prompt_by_run[rid] = _prompt_total(r)
    _assert_scored_ids({rid: recs for lst in h1b_by_device.values() for rid, recs in lst},
                       expected_ids, "H1b")
    # completeness: 2 clean cold + 3 clean warm bi transfer restarts per required device
    required_devices = [d for d in CONFIRMATORY_DEVICES
                        if d != CROSS_STRATUM_DEVICE or h1b_by_device.get(d) or not separate_runs]
    if separate_runs and not h1b_by_device.get(CROSS_STRATUM_DEVICE):
        required_devices = [d for d in CONFIRMATORY_DEVICES if d != CROSS_STRATUM_DEVICE]
    missing_runs = {}
    for dev in required_devices:
        cold = sum(1 for rid, _ in h1b_by_device.get(dev, [])
                   if (runs[rid]["meta"].get("cache") or "warm") == "cold")
        warm = sum(1 for rid, _ in h1b_by_device.get(dev, [])
                   if (runs[rid]["meta"].get("cache") or "warm") == "warm")
        if cold != 2 or warm != 3:
            missing_runs[dev] = {"cold_present": cold, "cold_required": 2,
                                 "warm_present": warm, "warm_required": 3,
                                 "missing": ([f"cold x{2 - cold}"] if cold < 2 else []) +
                                            ([f"warm x{3 - warm}"] if warm < 3 else [])}
    cross_stratum_pairs = []
    for rid, recs, dev, dg in separate_runs:
        for (orid, orecs), odev in [((rr, rc), dd) for dd, lst in sorted(h1b_by_device.items()) for rr, rc in lst]:
            b = byte_identity(recs, orecs)
            cross_stratum_pairs.append({"devices": [dev, odev], "device": dev, "runs": [rid, orid],
                                        "passed": b["passed"], "mismatch_ids": b["mismatch_ids"],
                                        "image_digests": {dev: dg, odev: EXPECTED_IMAGE_DIGEST},
                                        "enters_overall_pass": False})
    if smoke_ids is None:
        smoke_path = ROOT / "configs" / "r1" / "smoke_items.txt"
        smoke_ids = smoke_path.read_text().split() if smoke_path.exists() else []
    result["h1b"] = _h1b(h1b_by_device, prompt_by_run, smoke_ids=smoke_ids,
                          required_devices=required_devices, missing_runs=missing_runs,
                          cross_stratum_pairs=cross_stratum_pairs)
    return result

def main(argv=None):
    ap = argparse.ArgumentParser(prog="r1_analysis.py")
    ap.add_argument("--manifest", default=str(ROOT / "configs" / "r1" / "runs.json"))
    ap.add_argument("--items", default=str(ROOT / "configs" / "r1" / "items" / "transfer_core.jsonl"))
    ap.add_argument("--results-root", default=str(ROOT / "results"))
    ap.add_argument("--name", default="main")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    items_by_id = {}
    if Path(a.items).exists():
        items_by_id = {json.loads(l)["id"]: json.loads(l) for l in Path(a.items).read_text().splitlines() if l.strip()}
    manifest = json.loads(Path(a.manifest).read_text()) if Path(a.manifest).exists() else {"runs": []}
    res = analyze_manifest(manifest, items_by_id, Path(a.results_root))
    out = Path(a.out) if a.out else ROOT / "results" / "analysis" / f"r1_{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
