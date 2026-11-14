"""Live-benchmark decay analysis (preregistered; see PREREG.md).

Primary corpus: LiveCodeBench public code-generation leaderboard data
(per-model, per-problem pass@1 with contest dates and difficulty).
Secondary corpus: LiveBench public HF snapshots (per-model, per-question
scores with question release dates).

Pure functions here; analyze.py does file I/O and packaging.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

TOLERANCE_PP = 3.0          # preregistered comparability tolerance (percentage points)
TAU_BAR = 0.8               # preregistered descriptive rank-stability bar
MIN_PROBLEMS_PER_WINDOW = 50
N_BOOT = 10_000
PLACEHOLDER_RELEASE = dt.date(2023, 4, 30)  # LCB default cutoff marker; not a real model date
DIFFICULTIES = ("easy", "medium", "hard")


def quarter_of(ms: int) -> str:
    d = dt.datetime.fromtimestamp(ms / 1000, dt.UTC).date()
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


@dataclass
class Matrix:
    """Scores[model, problem] in percentage points plus problem metadata."""
    models: list[str]
    problems: list[str]
    scores: np.ndarray          # shape (M, P), float64, pp in [0, 100]
    window: np.ndarray          # shape (P,), window label per problem (str objects)
    stratum: np.ndarray         # shape (P,), stratum label per problem (difficulty or task)


def lcb_matrix(perf: dict) -> tuple[Matrix, dict]:
    rows = perf["performances"]
    models = sorted({r["model"] for r in rows})
    meta: dict[str, tuple[str, str, int]] = {}
    for r in rows:
        meta.setdefault(r["question_id"], (quarter_of(r["date"]), r["difficulty"], r["date"]))
    problems = sorted(meta)
    mi = {m: i for i, m in enumerate(models)}
    pi = {p: j for j, p in enumerate(problems)}
    scores = np.full((len(models), len(problems)), np.nan)
    for r in rows:
        scores[mi[r["model"]], pi[r["question_id"]]] = float(r["pass@1"])
    if np.isnan(scores).any():
        raise ValueError("LCB matrix is incomplete: every model must have every problem")
    window = np.array([meta[p][0] for p in problems], dtype=object)
    stratum = np.array([meta[p][1] for p in problems], dtype=object)
    dates = {p: meta[p][2] for p in problems}
    releases = {m["model_repr"]: m["release_date"] for m in perf["models"]}
    return Matrix(models, problems, scores, window, stratum), {"dates": dates, "releases": releases}


def included_windows(mat: Matrix) -> list[str]:
    labels, counts = np.unique(mat.window.astype(str), return_counts=True)
    keep = [str(lab) for lab, c in zip(labels, counts) if c >= MIN_PROBLEMS_PER_WINDOW]
    return sorted(keep)


def _poststratified_means(mat: Matrix, cols_by_window: dict[str, np.ndarray],
                          weights: dict[str, float]) -> dict[str, np.ndarray]:
    """Per-window model means reweighted to pooled stratum weights; NaN if a stratum is empty."""
    out = {}
    for w, cols in cols_by_window.items():
        total = np.zeros(len(mat.models))
        ok = True
        for s, wt in weights.items():
            sc = cols[mat.stratum[cols] == s]
            if wt == 0:
                continue
            if sc.size == 0:
                ok = False
                break
            total += wt * mat.scores[:, sc].mean(axis=1)
        out[w] = total if ok else np.full(len(mat.models), np.nan)
    return out


def drift_stats(mat: Matrix, windows: list[str], cols_by_window: dict[str, np.ndarray],
                weights: dict[str, float], model_mask: np.ndarray | None = None) -> dict:
    """Mean |drift| across models and adjacent window pairs, raw and adjusted, and Kendall tau."""
    mask = np.ones(len(mat.models), bool) if model_mask is None else model_mask
    raw = {w: mat.scores[:, cols_by_window[w]].mean(axis=1) for w in windows}
    adj = _poststratified_means(mat, {w: cols_by_window[w] for w in windows}, weights)
    raw_abs, adj_abs, taus = [], [], []
    for a, b in zip(windows, windows[1:]):
        raw_abs.append(np.abs(raw[b] - raw[a])[mask])
        if not (np.isnan(adj[a]).any() or np.isnan(adj[b]).any()):
            adj_abs.append(np.abs(adj[b] - adj[a])[mask])
        taus.append(kendall_tau_b(raw[a][mask], raw[b][mask]))
    raw_all = np.concatenate(raw_abs) if raw_abs else np.array([])
    adj_all = np.concatenate(adj_abs) if adj_abs else np.array([])
    return {
        "raw_mean_abs_drift": float(raw_all.mean()) if raw_all.size else math.nan,
        "adj_mean_abs_drift": float(adj_all.mean()) if adj_all.size else math.nan,
        "median_tau": float(np.median(taus)) if taus else math.nan,
        "n_cells_raw": int(raw_all.size),
        "n_cells_adj": int(adj_all.size),
        "pairs": len(windows) - 1,
    }


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        dx = np.sign(x[i + 1:] - x[i])
        dy = np.sign(y[i + 1:] - y[i])
        prod = dx * dy
        conc += int((prod > 0).sum())
        disc += int((prod < 0).sum())
        tx += int(((dx == 0) & (dy != 0)).sum())
        ty += int(((dy == 0) & (dx != 0)).sum())
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom else math.nan


def pooled_weights(mat: Matrix, cols: np.ndarray) -> dict[str, float]:
    labels, counts = np.unique(mat.stratum[cols].astype(str), return_counts=True)
    return {str(lab): c / counts.sum() for lab, c in zip(labels, counts)}


def analyze_corpus(mat: Matrix, windows: list[str], seed: int, n_boot: int = N_BOOT,
                   model_mask: np.ndarray | None = None) -> dict:
    cols_by_window = {w: np.flatnonzero(mat.window.astype(str) == w) for w in windows}
    all_cols = np.concatenate([cols_by_window[w] for w in windows])
    weights = pooled_weights(mat, all_cols)
    point = drift_stats(mat, windows, cols_by_window, weights, model_mask)
    rng = np.random.default_rng(seed)
    boot = {"raw_mean_abs_drift": [], "adj_mean_abs_drift": [], "median_tau": []}
    for _ in range(n_boot):
        resampled = {w: rng.choice(c, size=c.size, replace=True) for w, c in cols_by_window.items()}
        s = drift_stats(mat, windows, resampled, weights, model_mask)
        for k in boot:
            boot[k].append(s[k])
    ci = {}
    for k, vals in boot.items():
        arr = np.array(vals, float)
        arr = arr[~np.isnan(arr)]
        ci[k] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if arr.size else [math.nan, math.nan]
        ci[k + "_valid_resamples"] = int(arr.size)
    return {"point": point, "ci95": ci, "weights": weights, "windows": windows,
            "window_sizes": {w: int(c.size) for w, c in cols_by_window.items()}, "n_models": int(
                (model_mask if model_mask is not None else np.ones(len(mat.models), bool)).sum()),
            "n_boot": n_boot, "seed": seed}


def classify(ci_adj: list[float]) -> str:
    lo, hi = ci_adj
    if math.isnan(lo) or math.isnan(hi):
        return "inconclusive"
    if hi < TOLERANCE_PP:
        return "comparable"
    if lo > TOLERANCE_PP:
        return "not comparable"
    return "inconclusive"


def contamination_exploratory(mat: Matrix, extra: dict, seed: int, n_boot: int = 2000) -> list[dict]:
    """Exploratory only: post- minus pre-release pass@1 for models with a non-placeholder date."""
    out = []
    rng = np.random.default_rng(seed + 1)
    dates = np.array([extra["dates"][p] for p in mat.problems])
    for i, m in enumerate(mat.models):
        rel = extra["releases"].get(m)
        if rel is None:
            continue
        rel_date = dt.datetime.fromtimestamp(rel / 1000, dt.UTC).date()
        if rel_date == PLACEHOLDER_RELEASE:
            continue
        before = np.flatnonzero(dates < rel)
        after = np.flatnonzero(dates >= rel)
        if before.size < 30 or after.size < 30:
            continue

        def gap(b, a):
            w = pooled_weights(mat, np.concatenate([b, a]))
            vals = []
            for idx in (a, b):
                tot = 0.0
                for s, wt in w.items():
                    sc = idx[mat.stratum[idx] == s]
                    if sc.size == 0:
                        return math.nan
                    tot += wt * mat.scores[i, sc].mean()
                vals.append(tot)
            return vals[0] - vals[1]

        point = gap(before, after)
        boots = [gap(rng.choice(before, before.size), rng.choice(after, after.size)) for _ in range(n_boot)]
        arr = np.array(boots)
        arr = arr[~np.isnan(arr)]
        out.append({"model": m, "release_marker": rel_date.isoformat(), "n_before": int(before.size),
                    "n_after": int(after.size), "post_minus_pre_pp": point,
                    "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]})
    return out


def livebench_matrix(judgments: list[dict], release_index: dict[str, tuple[str, str, str]],
                     category: str) -> tuple[Matrix, dict]:
    """Build a model x question matrix for one category; NaN where a model lacks a question.

    release_index maps question_id -> (category, task, release_date).
    """
    rows = [r for r in judgments if r["question_id"] in release_index
            and release_index[r["question_id"]][0] == category and int(r.get("turn", 1)) == 1]
    models = sorted({r["model"] for r in rows})
    qids = sorted({r["question_id"] for r in rows})
    mi = {m: i for i, m in enumerate(models)}
    qi = {q: j for j, q in enumerate(qids)}
    scores = np.full((len(models), len(qids)), np.nan)
    for r in rows:
        scores[mi[r["model"]], qi[r["question_id"]]] = 100.0 * float(r["score"])
    window = np.array([release_index[q][2] for q in qids], dtype=object)
    stratum = np.array([release_index[q][1] for q in qids], dtype=object)
    return Matrix(models, qids, scores, window, stratum), {}


def livebench_pairs(mat: Matrix, seed: int, n_boot: int = N_BOOT) -> list[dict]:
    """Release-to-release drift per adjacent release pair, models complete on both, task-matched."""
    releases = sorted({str(w) for w in mat.window})
    results = []
    for a, b in zip(releases, releases[1:]):
        ca = np.flatnonzero(mat.window.astype(str) == a)
        cb = np.flatnonzero(mat.window.astype(str) == b)
        shared_tasks = sorted(set(mat.stratum[ca].astype(str)) & set(mat.stratum[cb].astype(str)))
        ca = ca[np.isin(mat.stratum[ca].astype(str), shared_tasks)]
        cb = cb[np.isin(mat.stratum[cb].astype(str), shared_tasks)]
        complete = ~np.isnan(mat.scores[:, np.concatenate([ca, cb])]).any(axis=1)
        if complete.sum() < 5 or ca.size < 10 or cb.size < 10:
            results.append({"pair": [a, b], "skipped": "fewer than 5 complete models or 10 task-matched questions",
                            "n_models": int(complete.sum()), "n_questions": [int(ca.size), int(cb.size)]})
            continue
        sub = Matrix([m for m, k in zip(mat.models, complete) if k], mat.problems,
                     mat.scores[complete], mat.window, mat.stratum)
        cols = {a: ca, b: cb}
        weights = pooled_weights(sub, np.concatenate([ca, cb]))
        point = drift_stats(sub, [a, b], cols, weights)
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_boot):
            rs = {a: rng.choice(ca, ca.size), b: rng.choice(cb, cb.size)}
            boot.append(drift_stats(sub, [a, b], rs, weights)["adj_mean_abs_drift"])
        arr = np.array(boot)
        arr = arr[~np.isnan(arr)]
        res = {"pair": [a, b], "tasks": shared_tasks, "n_models": int(complete.sum()),
               "n_questions": [int(ca.size), int(cb.size)], "point": point,
               "ci95_adj": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if arr.size else [math.nan, math.nan],
               "valid_resamples": int(arr.size)}
        res["classification"] = classify(res["ci95_adj"])
        results.append(res)
    return results
