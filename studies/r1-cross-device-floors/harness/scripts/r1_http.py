"""R1 serving-path screen arms via paper 2's harness.generate (no reimplemented generation HTTP).

Builds the vLLM server argv + the harness.generate command, optionally runs them, and writes
run_meta.json containing the server argv and a /metrics snapshot (fetched with urllib only
for the metrics endpoint, never for generation).
"""
from __future__ import annotations
import argparse, json, subprocess, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def build_server_cmd(model_path, port=8000, prefix_caching=False, kv_cache_dtype=None,
                     batch_invariant=False, max_num_seqs=None, budget_note=""):
    env_prefix = "VLLM_BATCH_INVARIANT=1 " if batch_invariant else ""
    cmd = [f"{env_prefix}vllm", "serve", str(model_path), "--port", str(port),
           "--dtype", "bfloat16", "--seed", "0"]
    if prefix_caching:
        cmd.append("--enable-prefix-caching")
    if kv_cache_dtype:
        cmd += ["--kv-cache-dtype", kv_cache_dtype]
    if max_num_seqs:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    return cmd

def build_generate_cmd(endpoint, model, run_id, step="r1", concurrency=96, benchmarks=None,
                       max_tokens_override=0, results_root=None, python=sys.executable):
    cmd = [python, "-m", "harness.generate", "--endpoint", endpoint, "--model", model,
           "--run-id", run_id, "--step", step, "--split", "dev",
           "--concurrency", str(concurrency)]
    if benchmarks:
        cmd += ["--benchmarks", benchmarks]
    if max_tokens_override:
        cmd += ["--max-tokens-override", str(max_tokens_override)]
    if results_root:
        cmd += ["--results-root", str(results_root)]
    return cmd

def fetch_metrics(endpoint_base: str) -> str:
    try:
        with urllib.request.urlopen(endpoint_base.rstrip("/") + "/metrics", timeout=15) as r:
            return r.read().decode("utf-8", "replace")[:20000]
    except Exception as e:
        return f"unavailable: {e}"

def write_run_meta(out_dir, server_cmd, generate_cmd, endpoint_base, extra=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {"server_argv": server_cmd, "generate_cmd": generate_cmd,
            "metrics_snapshot": fetch_metrics(endpoint_base) if endpoint_base else "",
            "extra": extra or {}}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

def main(argv=None):
    ap = argparse.ArgumentParser(prog="r1_http.py")
    ap.add_argument("--model-path", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--benchmarks", default="")
    ap.add_argument("--max-tokens-override", type=int, default=0)
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--kv-cache-dtype", default="")
    ap.add_argument("--batch-invariant", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    srv = build_server_cmd(a.model_path, a.port, a.prefix_caching, a.kv_cache_dtype or None, a.batch_invariant)
    gen = build_generate_cmd(a.endpoint, a.model, a.run_id, concurrency=a.concurrency,
                             benchmarks=a.benchmarks or None, max_tokens_override=a.max_tokens_override)
    if a.dry_run:
        print("SERVER:", " ".join(srv)); print("GENERATE:", " ".join(gen)); return 0
    write_run_meta(a.out, srv, gen, a.endpoint.rsplit("/v1", 1)[0],
                   {"concurrency": a.concurrency, "prefix_caching": a.prefix_caching,
                    "kv_cache_dtype": a.kv_cache_dtype or None})
    subprocess.run(gen, cwd=ROOT, check=False)
    write_run_meta(a.out, srv, gen, a.endpoint.rsplit("/v1", 1)[0],
                   {"concurrency": a.concurrency, "note": "metrics re-snapshotted after generate"})
    return 0

if __name__ == "__main__":
    sys.exit(main())
