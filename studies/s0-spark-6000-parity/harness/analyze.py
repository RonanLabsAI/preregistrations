#!/usr/bin/env python3
"""S0 analysis: byte identity between unit outputs, with treatment checks. Frozen before any run.

Stdlib only. Usage (run score.py first, then this once):
  python3 analyze.py --prompts prompts.jsonl \
      --unit q17-spark-c128=<unit receipts dir> ... (all 27 units) \
      [--ref oct-gb300-a=<harness.generate run dir> --ref oct-spark1-a=<dir>] \
      [--scores <unit>=<items.json from score.py> ...] --out analysis.json

A unit's receipts dir holds results.jsonl, treatment-a<N>.json and timing-a<N>.json (written by
s0_job.py) and serve.log (written by the unit harness). Two outputs are identical if both are
strings and byte-equal in UTF-8, matched by item id. The expected treatment of every unit, the
comparisons, the checks and the verdict rules are fixed here and in protocol-v1-2026-11-28.md.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

PROMPTS_SHA256 = "1d0ef52f6e1f6a158c6e6df87da13a8b2a298c1f6df450efc5daa8ba97997933"
N_PROMPTS = 1179
BI_IMAGE = "sha256:c92aec7952df6db14f57b021b8fe6bd91ff80919debfeeb14d2fc304e8549318"
BASE_IMAGE = "sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063"
BASE_ENV = sorted([
    "VLLM_BUILD_COMMIT=af7f9488c2210d67e1033ecdc845b087ee7fe92b",
    "VLLM_BUILD_PIPELINE=019d130e-464e-4ff7-b84b-492992c0c06b",
    "VLLM_BUILD_URL=https://buildkite.com/vllm/release-v2/builds/7359",
    "VLLM_ENABLE_CUDA_COMPATIBILITY=0",
    "VLLM_IMAGE_TAG=vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b",
    "VLLM_USAGE_SOURCE=production-docker-image",
])
BI_ENV = sorted(BASE_ENV + ["VLLM_BATCH_INVARIANT=1"])
SERVED = {"q17": "qwen3-1.7b", "g3": "granite-4.1-3b", "g8": "granite-4.1-8b"}
PLATFORM = {
    "spark": {"name": "NVIDIA GB10", "driver": "580.142", "compute_cap": "12.1"},
    "6000": {"name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "driver": "595.99.02",
             "compute_cap": "12.0"},
}
COMMON = ["--dtype", "bfloat16", "--max-model-len", "8192", "--seed", "0",
          "--max-num-batched-tokens", "8192", "--max-num-seqs", "1024", "--attention-backend", "FLASH_ATTN"]
MEM = {"kv16g": ["--gpu-memory-utilization", "0.25", "--kv-cache-memory-bytes", "17179869184"],
       "gmu085": ["--gpu-memory-utilization", "0.85"]}
MODELS = ["q17", "g3", "g8"]
REF_SHA256 = {
    "oct-gb300-a": {"gen/math/gsm8k.dev.jsonl": "bcbc7d71c9e2d1cb132148b7611a362f7c9c05c859bf013a2ac2ec30dd0149d6",
                    "gen/math/math500.dev.jsonl": "0395e5679b8f5770807aa150cd431d6f8aad8fa8e6c1fdcee5ada0e5c6f6b3cb",
                    "gen/general/ifeval.dev.jsonl": "6f6b8df6c32d1d726cef757c2add57e13127cd39840688f71034fed389e59c30"},
    "oct-spark1-a": {"gen/math/gsm8k.dev.jsonl": "a5b3d78dd1a387724e05d59913b603ea993d8871431647e637b37b86998d9f63",
                     "gen/math/math500.dev.jsonl": "0ff4acbfa1308f1735df9f547be13bfe33fb6fa76975e13aed4ae30ede19afbe",
                     "gen/general/ifeval.dev.jsonl": "42f1060b517298f57a57de04bff8149d152392f3001f22a3fbb3eacecdb29a13"},
}


def expected_units():
    """unit -> (model, hardware, batch_invariant, concurrency, memory mode)"""
    e = {}
    for m in MODELS:
        e[f"{m}-spark-c128"] = (m, "spark", True, 128, "kv16g")
        e[f"{m}-spark-c128-r2"] = (m, "spark", True, 128, "kv16g")
        e[f"{m}-spark-c64"] = (m, "spark", True, 64, "kv16g")
        e[f"{m}-6000-c128"] = (m, "6000", True, 128, "kv16g")
        e[f"{m}-6000-c128-r2"] = (m, "6000", True, 128, "kv16g")
        e[f"{m}-6000-c32"] = (m, "6000", True, 32, "kv16g")
        e[f"{m}-6000-gmu085-c128"] = (m, "6000", True, 128, "gmu085")
        e[f"{m}-6000-nobi-c128"] = (m, "6000", False, 128, "kv16g")
        e[f"{m}-6000-nobi-c32"] = (m, "6000", False, 32, "kv16g")
    return e


EXPECT = expected_units()
BATCH_COMPONENTS = {"batch_spark", "batch_6000", "control_bi_off"}


def components(m):
    return {
        "hardware": (f"{m}-spark-c128", f"{m}-6000-c128"),
        "determinism_spark": (f"{m}-spark-c128", f"{m}-spark-c128-r2"),
        "determinism_6000": (f"{m}-6000-c128", f"{m}-6000-c128-r2"),
        "batch_spark": (f"{m}-spark-c128", f"{m}-spark-c64"),
        "batch_6000": (f"{m}-6000-c128", f"{m}-6000-c32"),
        "control_bi_off": (f"{m}-6000-nobi-c128", f"{m}-6000-nobi-c32"),
    }


def reported(m):
    return {
        "kv_pool_6000 (V1 deployment flags, one run)": (f"{m}-6000-c128", f"{m}-6000-gmu085-c128"),
        "hardware at unequal lower concurrencies (Spark 64, RTX PRO 6000 32)": (f"{m}-spark-c64", f"{m}-6000-c32"),
        "hardware, replicates": (f"{m}-spark-c128-r2", f"{m}-6000-c128-r2"),
    }


REFERENCE = [("q17-spark-c128", "oct-spark1-a", "Spark now vs Spark 2026-10-29"),
             ("q17-spark-c128", "oct-gb300-a", "Spark now vs GB300 2026-10-29"),
             ("q17-6000-c128", "oct-gb300-a", "6000 now vs GB300 2026-10-29")]


def parse_jsonl(path):
    rows, bad = [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(r, dict):
            rows.append(r)
        else:
            bad += 1
    return rows, bad


def strip_model_and_port(args):
    """Container args minus the image's `serve` subcommand and host-specific values (model path, host,
    port). The harness passes --model, --served-model-name, --host, --port, then the spec's serve_args."""
    out, skip = [], False
    if args[:1] == ["serve"]:
        args = args[1:]
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--model", "--port", "--host"):
            skip = True
            continue
        out.append(a)
    return out


def check_treatment(name, rdir):
    """Problems with what the unit actually ran, from every treatment-a<N>.json."""
    m, hw, bi, conc, mem = EXPECT[name]
    want_args = ["--served-model-name", SERVED[m]] + COMMON + MEM[mem]
    plat = PLATFORM[hw]
    problems, uuids, attempts = [], set(), set()
    files = sorted(Path(rdir).glob("treatment-a*.json"))
    if not files:
        return ["no treatment record"], uuids, attempts
    for f in files:
        tag = f.name
        try:
            t = json.load(open(f, encoding="utf-8"))
        except ValueError:
            problems.append(f"{tag}: unreadable")
            continue
        attempts.add(t.get("attempt"))
        if t.get("unit") != name:
            problems.append(f"{tag}: unit label {t.get('unit')!r}")
        if t.get("concurrency") != conc:
            problems.append(f"{tag}: concurrency {t.get('concurrency')!r}")
        if t.get("machine") != "aarch64":
            problems.append(f"{tag}: machine {t.get('machine')!r}")
        gpu = t.get("gpu") or {}
        for key in ("name", "driver", "compute_cap"):
            if gpu.get(key) != plat[key]:
                problems.append(f"{tag}: GPU {key} {gpu.get(key)!r}")
        if not gpu.get("uuid"):
            problems.append(f"{tag}: no GPU UUID")
        uuids.add(gpu.get("uuid"))
        if t.get("image_id") != (BI_IMAGE if bi else BASE_IMAGE):
            problems.append(f"{tag}: image {t.get('image_id')!r}")
        if sorted(t.get("vllm_env") or []) != (BI_ENV if bi else BASE_ENV):
            problems.append(f"{tag}: VLLM_* environment {t.get('vllm_env')!r}")
        if strip_model_and_port(t.get("args") or []) != want_args:
            problems.append(f"{tag}: serve args {strip_model_and_port(t.get('args') or [])!r}")
        served = [d.get("id") for d in (t.get("server_models") or {}).get("data", [])]
        if served != [SERVED[m]]:
            problems.append(f"{tag}: served models {served!r}")
    return problems, uuids, attempts


def server_log_facts(rdir):
    """Attention backend, KV cache size and running-request counts from the harness's serve.log."""
    p = Path(rdir) / "serve.log"
    if not p.exists():
        return {"present": False}
    text = p.read_text(encoding="utf-8", errors="replace")
    backends = sorted(set(re.findall(r"Using (?:AttentionBackendEnum\.)?([A-Z_]+) (?:attention )?backend", text)))
    kv = sorted(set(k.replace(",", "") for k in re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)))
    running = [int(x) for x in re.findall(r"Running: (\d+) reqs", text)]
    # Idle samples (Running: 0) are left out, so overlap means both devices ran comparable batches.
    hist = collections.Counter(min((r - 1) // 16 * 16 + 1, 1025) for r in running if r > 0)
    return {"present": True, "attention_backends": backends, "kv_cache_tokens": kv,
            "max_running_reqs": max(running) if running else None,
            "running_reqs_histogram_16": {str(k): v for k, v in sorted(hist.items())}}


def wall_seconds(rdir):
    total = 0.0
    for f in sorted(Path(rdir).glob("timing-a*.json")):
        try:
            t = json.load(open(f, encoding="utf-8"))
            total += t["end_unix"] - t["start_unix"]
        except (ValueError, KeyError, TypeError):
            pass
    return round(total, 1)


def load_unit(name, rdir, ids):
    path = Path(rdir) / "results.jsonl"
    info = {"receipts": str(rdir), "problems": []}
    if not path.exists():
        info["problems"].append("no results.jsonl")
        return {}, info
    info["results_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, bad = parse_jsonl(path)
    by_id, dup, conflicting, schema = {}, 0, 0, 0
    for r in rows:
        i = r.get("id")
        if i in by_id:
            dup += 1
            if r.get("output") != by_id[i].get("output"):
                conflicting += 1
            continue  # first record wins; duplicates are counted and reported
        by_id[i] = r
        if not (isinstance(r.get("output"), str) and r.get("finish_reason") in ("stop", "length")
                and isinstance(r.get("n_out_tokens"), int) and isinstance(r.get("n_prompt_tokens"), int)):
            schema += 1
    extra = sorted(set(by_id) - set(ids), key=str)
    missing = sorted(set(ids) - set(by_id))
    tproblems, uuids, t_attempts = check_treatment(name, rdir)
    result_attempts = {r.get("attempt") for r in by_id.values()}
    log = server_log_facts(rdir)
    info.update({
        "records": len(by_id), "unparseable_lines": bad, "duplicates": dup,
        "conflicting_duplicates": conflicting, "schema_errors": schema,
        "missing_ids": len(missing), "unknown_ids": len(extra),
        "gpu_uuids": sorted(u for u in uuids if u),
        "finish_reason": dict(collections.Counter(str(r.get("finish_reason")) for r in by_id.values())),
        "attempts": dict(collections.Counter(str(r.get("attempt")) for r in by_id.values())),
        "out_tokens_total": sum(r["n_out_tokens"] for r in by_id.values() if isinstance(r.get("n_out_tokens"), int)),
        "wall_seconds": wall_seconds(rdir),
        "server_log": log,
    })
    p = info["problems"]
    p += tproblems
    if missing or extra or schema:
        p.append(f"missing {len(missing)}, unknown {len(extra)}, schema errors {schema}")
    if conflicting:
        p.append(f"{conflicting} conflicting duplicate record(s)")
    if result_attempts - t_attempts:
        p.append(f"results from attempts without a treatment record: {sorted(result_attempts - t_attempts, key=str)}")
    if not log["present"]:
        p.append("no serve.log")
    else:
        if log["attention_backends"] != ["FLASH_ATTN"]:
            p.append(f"attention backends in serve.log {log['attention_backends']!r}")
        if len(log["kv_cache_tokens"]) != 1:
            p.append(f"KV cache size in serve.log {log['kv_cache_tokens']!r}")
        if log["max_running_reqs"] is None:
            p.append("no running-request counts in serve.log")
    return by_id, info


def load_ref(name, rdir):
    """Reference run outputs, after checking every file against its frozen SHA-256."""
    by_id = {}
    for rel, want in REF_SHA256[name].items():
        f = Path(rdir) / rel
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != want:
            raise ValueError(f"{name}: {rel} missing or SHA-256 mismatch")
        rows, _ = parse_jsonl(f)
        for r in rows:
            if "id" in r:
                by_id.setdefault(r["id"], r)
    return by_id


def first_diff(a, b):
    ab, bb = a.encode("utf-8"), b.encode("utf-8")
    for i in range(min(len(ab), len(bb))):
        if ab[i] != bb[i]:
            return i
    return min(len(ab), len(bb))


def compare(a, b, ids, bench_of, sa=None, sb=None, check_prompt_tokens=True):
    res = {"n": len(ids), "identical": 0, "missing": 0, "prompt_token_disagree": 0,
           "finish_reason_disagree": 0, "by_bench": {}, "by_attempt": {}, "mismatches": []}
    per = collections.defaultdict(lambda: [0, 0])
    by_att = collections.defaultdict(lambda: [0, 0])
    disc = collections.defaultdict(collections.Counter)
    for i in ids:
        ra, rb = a.get(i), b.get(i)
        if ra is None or rb is None or not isinstance(ra.get("output"), str) or not isinstance(rb.get("output"), str):
            res["missing"] += 1
            continue
        if check_prompt_tokens and ra.get("n_prompt_tokens") != rb.get("n_prompt_tokens"):
            res["prompt_token_disagree"] += 1
        bench, att = bench_of[i], f"{ra.get('attempt')}/{rb.get('attempt')}"
        per[bench][1] += 1
        by_att[att][1] += 1
        if ra["output"] == rb["output"]:
            res["identical"] += 1
            per[bench][0] += 1
            by_att[att][0] += 1
        else:
            res["mismatches"].append({
                "id": i, "bench": bench, "first_diff_byte": first_diff(ra["output"], rb["output"]),
                "len_a": len(ra["output"].encode("utf-8")), "len_b": len(rb["output"].encode("utf-8")),
                "tok_a": ra.get("n_out_tokens"), "tok_b": rb.get("n_out_tokens"),
                "finish_a": ra.get("finish_reason"), "finish_b": rb.get("finish_reason"),
                "attempt_a": ra.get("attempt"), "attempt_b": rb.get("attempt")})
        if ra.get("finish_reason") != rb.get("finish_reason"):
            res["finish_reason_disagree"] += 1
        if sa is not None and sb is not None and i in sa and i in sb:
            disc[bench][(bool(sa[i]), bool(sb[i]))] += 1
    res["by_bench"] = {k: {"identical": v[0], "n": v[1]} for k, v in sorted(per.items())}
    res["by_attempt"] = {k: {"identical": v[0], "n": v[1]} for k, v in sorted(by_att.items())}
    if disc:
        scores = {}
        for bench, c in sorted(disc.items()):
            n = sum(c.values())
            acc_a = 100.0 * (c[(True, True)] + c[(True, False)]) / n
            acc_b = 100.0 * (c[(True, True)] + c[(False, True)]) / n
            scores[bench] = {"n": n, "both": c[(True, True)], "neither": c[(False, False)],
                             "a_only": c[(True, False)], "b_only": c[(False, True)],
                             "score_a": round(acc_a, 4), "score_b": round(acc_b, 4),
                             "score_diff_b_minus_a": round(acc_b - acc_a, 4)}
        res["scores_by_bench"] = scores
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--unit", action="append", default=[])
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--scores", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    raw = Path(args.prompts).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROMPTS_SHA256:
        sys.exit("prompts.jsonl does not match the frozen SHA-256")
    prompts = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    ids = [p["id"] for p in prompts]
    if len(ids) != N_PROMPTS or len(set(ids)) != N_PROMPTS:
        sys.exit("prompt count or ids do not match the protocol")
    bench_of = {p["id"]: p["bench"] for p in prompts}

    errors, runs, units = [], {}, {}
    for spec in args.unit:
        name, path = spec.split("=", 1)
        if name not in EXPECT:
            errors.append(f"unknown unit {name} ignored")
            continue
        try:
            runs[name], units[name] = load_unit(name, path, ids)
        except (OSError, UnicodeDecodeError) as e:
            runs[name], units[name] = {}, {"receipts": path, "problems": [f"unreadable: {e}"]}
    refs = {}
    for spec in args.ref:
        name, path = spec.split("=", 1)
        try:
            if name not in REF_SHA256:
                raise ValueError(f"unknown reference {name}")
            refs[name] = load_ref(name, path)
        except (OSError, ValueError) as e:
            errors.append(f"reference {name} not used: {e}")
    scores = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        try:
            s = json.load(open(path, encoding="utf-8"))
            if not isinstance(s, dict):
                raise ValueError("not an object")
            scores[name] = s
        except (OSError, ValueError) as e:
            errors.append(f"scores for {name} not used: {e}")

    def kv_tokens(u):
        return units[u].get("server_log", {}).get("kv_cache_tokens")

    def max_running(u):
        return units[u].get("server_log", {}).get("max_running_reqs")

    def comp(a, b, label):
        if a not in runs or b not in runs:
            return {"a": a, "b": b, "label": label, "status": "INCOMPLETE", "why": ["unit not supplied"]}
        why = [f"{u}: {p}" for u in (a, b) for p in units[u]["problems"]]
        r = compare(runs[a], runs[b], ids, bench_of, scores.get(a), scores.get(b))
        if r["missing"] or r["prompt_token_disagree"]:
            why.append(f"missing {r['missing']}, prompt-token disagreements {r['prompt_token_disagree']}")
        if EXPECT[a][4] == EXPECT[b][4] == "kv16g" and kv_tokens(a) != kv_tokens(b):
            why.append(f"KV cache sizes differ: {kv_tokens(a)} vs {kv_tokens(b)}")
        if label == "hardware":
            ha = set(units[a].get("server_log", {}).get("running_reqs_histogram_16", {}))
            hb = set(units[b].get("server_log", {}).get("running_reqs_histogram_16", {}))
            if not (ha & hb):
                why.append("running-request histograms of the two devices do not overlap")
        if label in BATCH_COMPONENTS:
            lo, hi = (b, a) if EXPECT[b][3] < EXPECT[a][3] else (a, b)
            if not (isinstance(max_running(lo), int) and isinstance(max_running(hi), int)
                    and max_running(lo) < max_running(hi)):
                why.append(f"server batches did not differ: max running {max_running(lo)} vs {max_running(hi)}")
        status = "INCOMPLETE" if why else ("IDENTICAL" if r["identical"] == len(ids) else "DIFFERENT")
        return {"a": a, "b": b, "label": label, "status": status, "why": why, **r}

    def same_card(hw, m):
        names = [u for u, e in EXPECT.items() if e[0] == m and e[1] == hw]
        if not all(u in units for u in names):
            return False
        cards = [units[u].get("gpu_uuids", []) for u in names]
        return all(len(c) == 1 for c in cards) and len({c[0] for c in cards}) == 1

    out = {"prompts": len(ids), "errors": errors, "units": units, "models": {}, "reference": []}
    for m in MODELS:
        comps = {k: comp(a, b, k) for k, (a, b) in components(m).items()}
        flags = []
        for hw in ("spark", "6000"):
            if not same_card(hw, m):
                flags.append(f"{hw} units of this model did not all report one and the same GPU UUID")
        st = {k: c["status"] for k, c in comps.items()}
        core = [k for k in comps if k != "control_bi_off"]
        if flags or "INCOMPLETE" in st.values():
            verdict = "INCOMPLETE"
        elif "DIFFERENT" in (st["determinism_spark"], st["determinism_6000"]):
            verdict = "NONDETERMINISTIC"
        elif "DIFFERENT" in (st["batch_spark"], st["batch_6000"]):
            verdict = "BATCH-VARIANT"
        elif st["hardware"] == "DIFFERENT":
            verdict = "HARDWARE-DIFFERENT"
        elif st["control_bi_off"] == "IDENTICAL":
            verdict = "UNINFORMATIVE"  # every core comparison identical, but the control shows no noise
        elif all(st[k] == "IDENTICAL" for k in core):
            verdict = "PASS"
        else:
            verdict = "INCOMPLETE"
        rep = {k: comp(a, b, k) for k, (a, b) in reported(m).items()}
        out["models"][m] = {"verdict": verdict, "pooling_allowed": verdict == "PASS",
                            "flags": flags, "components": comps, "reported": rep}
    for a, b, label in REFERENCE:
        if a in runs and b in refs:
            r = compare(runs[a], refs[b], ids, bench_of, check_prompt_tokens=False)
            out["reference"].append({"a": a, "b": b, "label": label, **r})

    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for e in errors:
        print(f"note: {e}")
    for m in MODELS:
        v = out["models"][m]
        line = "; ".join(f"{k}: {c['status']} {c.get('identical', '-')}/{c.get('n', '-')}"
                         for k, c in v["components"].items())
        print(f"{m}: {v['verdict']} ({line})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
