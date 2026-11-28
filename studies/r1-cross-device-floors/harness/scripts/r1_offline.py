"""R1 scheduler-matched offline arm; runs INSIDE the vLLM container.

vllm is imported lazily inside run() so tests/host python without vllm can import this module.
Sets VLLM_BATCH_INVARIANT / VLLM_ATTENTION_BACKEND in the environment BEFORE importing vllm,
TF32 off, SamplingParams(temperature=0, seed=0). Output in paper-2 layout
gen/<domain>/<bench>.dev.jsonl with both `text` (card) and `output` (harness.aggregate alias),
plus run_meta.json. Fails the run if a tuned-config dir is in effect (vLLM #70406).
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_DOMAIN = {"gsm8k": "math", "math500": "math", "ifeval": "general", "mmlu_pro": "general",
                "livecodebench": "code", "humaneval_plus": "code", "mbpp_plus": "code",
                "bird_dev": "sql", "bfcl": "tools", "flores200": "translate"}

BLOCK_SIZE = 16
KV_CACHE_DTYPE = "bfloat16"

def build_engine_kwargs(model, mode="default", num_gpu_blocks=None, attention_backend=None,
                        max_num_seqs=32, max_num_batched_tokens=8192, max_model_len=None,
                        block_size=BLOCK_SIZE, kv_cache_dtype=KV_CACHE_DTYPE):
    kw = {"model": str(model), "enable_chunked_prefill": True, "enable_prefix_caching": False,
          "dtype": "bfloat16", "kv_cache_dtype": kv_cache_dtype, "block_size": int(block_size), "seed": 0,
          "max_num_seqs": max_num_seqs, "max_num_batched_tokens": max_num_batched_tokens,
          "enforce_eager": False, "disable_log_stats": False}  # engine stats on: preemptions for the scheduler-match check
    if num_gpu_blocks is not None:
        kw["num_gpu_blocks_override"] = int(num_gpu_blocks)
    if attention_backend:
        kw["attention_config"] = {"backend": attention_backend}
    if max_model_len:
        kw["max_model_len"] = int(max_model_len)
    return kw

def env_for_mode(mode, attention_backend=None):
    env = {"VLLM_BATCH_INVARIANT": "1" if mode == "bi" else "0", "NVIDIA_TF32_OVERRIDE": "0"}
    if attention_backend:
        env["VLLM_ATTENTION_BACKEND"] = attention_backend
    return env

def permuted_order(ids, seed=0):
    ids = list(ids)
    if not seed:
        return ids
    import random
    out = list(ids)
    random.Random(seed).shuffle(out)
    return out

def canonical_order(records, canonical_ids):
    pos = {i: k for k, i in enumerate(canonical_ids)}
    return sorted(records, key=lambda r: pos.get(r["id"], 1 << 30))

def max_tokens_for(bench, default, caps=None):
    if caps:
        return int(caps.get(bench, caps.get("default", default)))
    return int(default)

def group_items(items):
    out = {}
    for it in items:
        bench = it.get("benchmark") or it.get("bench")
        if bench is None:
            raise KeyError(f"item {it.get('id')} has no benchmark field")
        out.setdefault((BENCH_DOMAIN.get(bench, "general"), bench), []).append(it)
    return out

_TUNED = re.compile(r"tuned[\s_\-]*config|tuned_config_dir|loading tuned|tuned config from", re.I)
_BACKEND = re.compile(r"(?:Using|attention backend[: =]+)\s*([A-Z0-9_\-]+(?:_ATTN|ATTENTION|FLASH_ATTN|TRITON_ATTN|FLASHINFER)?)", re.I)
_GRAPH = re.compile(r"captur(?:ing|ed) cuda graphs|graph capturing", re.I)

def parse_engine_log(text: str) -> dict:
    backend = None
    m = re.search(r"([A-Z][A-Z0-9_]*(?:_ATTN|ATTENTION))\b", text) or re.search(r"FLASH_ATTN|FLASHINFER|TRITON_ATTN", text)
    if m:
        backend = m.group(1) if m.lastindex else m.group(0)
    tuned_lines = [l for l in text.splitlines() if _TUNED.search(l) and "no tuned" not in l.lower()
                   and "not found" not in l.lower()]
    low = text.lower()
    if "breakable" in low:
        graph_mode = "breakable"
    elif _GRAPH.search(text) or "piecewise" in low:
        graph_mode = "piecewise"
    else:
        graph_mode = None
    return {"attention_backend": backend,
            "cuda_graphs": graph_mode is not None, "cuda_graph_mode": graph_mode,
            "cuda_graph_capture_state": "captured" if graph_mode else "not_observed_in_log",
            "tuned_config_in_effect": bool(tuned_lines), "tuned_config_lines": tuned_lines[:5]}

def assert_no_tuned_config(info: dict):
    if info.get("tuned_config_in_effect"):
        raise RuntimeError(f"tuned config in effect (vLLM #70406 overrides BI): {info.get('tuned_config_lines')}")

def expected_cudagraph_pair(mode) -> tuple:
    """Expected read-back (cudagraph_mode, breakable) per mode at pin 3f1bcfd8.

    Source (gh api, vllm/config/compilation.py): cudagraph_mode v1 default is
    FULL_AND_PIECEWISE; breakable is env VLLM_USE_BREAKABLE_CUDAGRAPH
    (vllm/compilation/breakable_cudagraph.is_breakable_cudagraph_enabled),
    auto-enabled under VLLM_BATCH_INVARIANT (vllm/config/vllm.py
    _maybe_enable_breakable_cudagraph). Live smoke: bi -> breakable True.
    """
    return ("FULL_AND_PIECEWISE", True) if mode == "bi" else ("FULL_AND_PIECEWISE", False)

def declared_cuda_graph_mode(mode) -> str:
    """Expected cudagraph_mode read-back (kept name for callers; no longer a log label)."""
    return expected_cudagraph_pair(mode)[0]

def assert_engine_log_ok(log_text, mode="default", attention_backend=None) -> dict:
    """Fail the run if the engine log is empty, lacks the backend line, or has a tuned
    config in effect. CUDA-graph mode is verified from the config read-back, never
    inferred from log lines (Deviation 4 #5)."""
    if isinstance(log_text, dict):
        info = log_text
        if not info.get("attention_backend"):
            raise RuntimeError("engine log lacks the attention backend line")
    else:
        if not log_text or not str(log_text).strip():
            raise RuntimeError("engine log is empty")
        info = parse_engine_log(str(log_text))
        if not info.get("attention_backend"):
            raise RuntimeError("engine log lacks the attention backend line")
    assert_no_tuned_config(info)
    if attention_backend and str(info.get("attention_backend")).upper() != str(attention_backend).upper():
        raise RuntimeError(f"attention backend mismatch: requested {attention_backend}, "
                           f"log has {info.get('attention_backend')}")
    return info

def _sub(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

def _norm_name(v):
    if v is None:
        return None
    name = getattr(v, "name", None) or getattr(v, "value", None) or str(v)
    return str(name).rsplit(".", 1)[-1].upper()

def _norm_dtype(v):
    if v is None:
        return None
    s = str(getattr(v, "name", None) or v).lower()
    return {"bf16": "bfloat16", "fp16": "float16"}.get(s, s)

def expected_engine_config(num_gpu_blocks=None, attention_backend=None, max_model_len=None,
                           max_num_seqs=32, max_num_batched_tokens=8192,
                           block_size=BLOCK_SIZE, kv_cache_dtype=KV_CACHE_DTYPE,
                           enable_prefix_caching=False, mode="default",
                           enable_chunked_prefill=True) -> dict:
    cg_mode, breakable = expected_cudagraph_pair(mode)
    return {"block_size": int(block_size), "num_gpu_blocks": num_gpu_blocks,
            "kv_cache_dtype": kv_cache_dtype, "max_num_seqs": int(max_num_seqs),
            "max_num_batched_tokens": int(max_num_batched_tokens), "max_model_len": max_model_len,
            "enable_prefix_caching": bool(enable_prefix_caching), "attention_backend": attention_backend,
            "enable_chunked_prefill": bool(enable_chunked_prefill),
            "cudagraph_mode": cg_mode, "breakable_cudagraph": bool(breakable)}

def read_engine_config(source) -> dict:
    """Read back the resolved engine config at the pin (vllm v0.32.0rc1.dev190).

    Attribute paths (vllm/config/{cache,scheduler,model,attention,compilation}.py,
    vllm/entrypoints/llm.py): llm.llm_engine.vllm_config.cache_config.{block_size,
    num_gpu_blocks (set after profiling), cache_dtype}; .scheduler_config.{max_num_seqs,
    max_num_batched_tokens}; .model_config.max_model_len; .cache_config.enable_prefix_caching;
    .attention_config.backend; .compilation_config.cudagraph_mode.
    """
    vc = source
    if _sub(vc, "llm_engine") is not None:
        vc = _sub(vc, "llm_engine")
    if _sub(vc, "vllm_config") is not None:
        vc = _sub(vc, "vllm_config")
    def _breakable(*objs):
        for o in objs:
            for n in ("use_breakable_cudagraph", "breakable_cudagraph", "enable_breakable_cudagraph",
                      "breakable", "is_breakable_cudagraph_enabled"):
                v = _sub(o, n)
                if v is not None:
                    return bool(v() if callable(v) else v)
        env = os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH")
        return (env == "1") if env is not None else False
    if _sub(vc, "cache_config") is None and any(_sub(vc, k) is not None for k in ("block_size", "kv_cache_dtype")):
        cg = _norm_name(_sub(vc, "cudagraph_mode") or _sub(vc, "cuda_graph_mode"))
        br = _breakable(vc)
        return {"block_size": _sub(vc, "block_size"),
                "num_gpu_blocks": _sub(vc, "num_gpu_blocks"),
                "kv_cache_dtype": _norm_dtype(_sub(vc, "kv_cache_dtype") or _sub(vc, "cache_dtype")),
                "max_num_seqs": _sub(vc, "max_num_seqs"),
                "max_num_batched_tokens": _sub(vc, "max_num_batched_tokens"),
                "max_model_len": _sub(vc, "max_model_len"),
                "enable_prefix_caching": _sub(vc, "enable_prefix_caching"),
                "enable_chunked_prefill": _sub(vc, "enable_chunked_prefill"),
                "attention_backend": _norm_name(_sub(vc, "attention_backend")),
                "cuda_graph_mode": cg, "cudagraph_mode": cg,
                "breakable_cudagraph": br, "use_breakable_cudagraph": br}
    cache = _sub(vc, "cache_config") or {}
    sched = _sub(vc, "scheduler_config") or {}
    model = _sub(vc, "model_config") or {}
    attn = _sub(vc, "attention_config") or {}
    comp = _sub(vc, "compilation_config") or {}
    num_blocks = _sub(cache, "num_gpu_blocks")
    if num_blocks is None:
        num_blocks = _sub(cache, "num_gpu_blocks_override")
    cg = _norm_name(_sub(comp, "cudagraph_mode"))
    br = _breakable(comp, vc)
    return {"block_size": _sub(cache, "block_size"), "num_gpu_blocks": num_blocks,
            "kv_cache_dtype": _norm_dtype(_sub(cache, "cache_dtype") or _sub(cache, "kv_cache_dtype")),
            "max_num_seqs": _sub(sched, "max_num_seqs"),
            "max_num_batched_tokens": _sub(sched, "max_num_batched_tokens"),
            "max_model_len": _sub(model, "max_model_len") or _sub(sched, "max_model_len"),
            "enable_prefix_caching": _sub(cache, "enable_prefix_caching"),
            "enable_chunked_prefill": _sub(sched, "enable_chunked_prefill"),
            "attention_backend": _norm_name(_sub(attn, "backend") or _sub(attn, "attention_backend")),
            "cuda_graph_mode": cg, "cudagraph_mode": cg,
            "breakable_cudagraph": br, "use_breakable_cudagraph": br}

def verify_engine_config(actual: dict, expected: dict) -> list:
    """List of mismatches between read-back (`actual`) and requested (`expected`); [] = match."""
    out = []
    for key, exp in (expected or {}).items():
        if exp is None:
            continue
        act = (actual or {}).get(key)
        if key == "attention_backend":
            same = _norm_name(act) == _norm_name(exp)
        elif key == "kv_cache_dtype":
            same = _norm_dtype(act) == _norm_dtype(exp)
        elif key in ("enable_prefix_caching", "enable_chunked_prefill",
                     "breakable_cudagraph", "use_breakable_cudagraph"):
            same = act is not None and bool(act) == bool(exp)
        elif key in ("cudagraph_mode", "cuda_graph_mode"):
            same = _norm_name(act) == _norm_name(exp)
        elif isinstance(exp, (int, float)) and not isinstance(exp, bool):
            try:
                same = act is not None and int(act) == int(exp)
            except Exception:
                same = False
        else:
            same = act == exp
        if not same:
            out.append(f"{key}: requested {exp!r}, got {act!r}")
    return out

def assert_engine_config_matches(actual: dict, expected: dict):
    mismatches = verify_engine_config(actual, expected)
    if mismatches:
        raise RuntimeError("engine config read-back mismatch: " + "; ".join(mismatches))
    return actual

def assert_prefix_cache_zero(stats: dict):
    hits = (stats or {}).get("prefix_cache_hits")
    if hits != 0:
        raise RuntimeError(f"prefix-cache hits must be 0, got {hits!r} (vllm:prefix_cache_hits)")

def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""

def env_fingerprint() -> dict:
    import platform
    fp = {"platform": platform.platform(), "arch": platform.machine(), "python": platform.python_version(),
          "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap",
                              "--format=csv,noheader"]),
          "pip_freeze_sha256": hashlib.sha256(_run([sys.executable, "-m", "pip", "freeze"]).encode()).hexdigest()}
    try:
        import torch
        fp.update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "cudnn": str(torch.backends.cudnn.version())})
    except Exception as e:
        fp["torch"] = f"unavailable: {e}"
    try:
        import vllm
        fp["vllm"] = getattr(vllm, "__version__", "unknown")
        fp["vllm_git_sha"] = _run(["git", "-C", str(Path(vllm.__file__).parent), "rev-parse", "HEAD"]) or "unknown"
    except Exception as e:
        fp["vllm"] = f"unavailable: {e}"
    return fp

def cache_listing(base) -> list:
    base = Path(base)
    return sorted(str(p.relative_to(base)) for p in base.rglob("*")) if base.exists() else []

def cache_listing_count(base) -> int:
    if isinstance(base, (int, float)):
        return int(base)
    if isinstance(base, (list, tuple)):
        return len(base)
    return len(cache_listing(base))

def assert_warm_cache_nonempty(cache, cache_dir, is_warmup=False):
    """Abort a --cache warm run whose cache dir listing at start is empty (warmup exempt)."""
    if cache != "warm" or is_warmup:
        return
    if cache_listing_count(cache_dir) == 0:
        raise RuntimeError(f"warm cache dir empty at start (n_entries_at_start=0): {cache_dir}")

def filter_items_by_ids(items, ids):
    """Filter items to the id list, preserving items-file order."""
    want = set(ids or [])
    return [it for it in items if it.get("id") in want]

def _cache_setup(cache, cache_dir, run_dir: Path):
    if cache == "cold":
        base = run_dir / "cold_cache"
        dirs = {k: base / k for k in ["vllm", "torchinductor", "triton", "torch"]}
    else:
        base = Path(cache_dir) if cache_dir else run_dir / "warm_cache"
        dirs = {k: base / k for k in ["vllm", "torchinductor", "triton", "torch"]}
    listing_before = cache_listing(base)  # before any mkdir / engine init
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    env = {"VLLM_CACHE_ROOT": str(dirs["vllm"]), "TORCHINDUCTOR_CACHE_DIR": str(dirs["torchinductor"]),
           "TRITON_CACHE_DIR": str(dirs["triton"]), "TORCH_COMPILE_CACHE_DIR": str(dirs["torch"])}
    listing = cache_listing(base)
    return env, {"mode": cache, "dir": str(base),
                 "listing_sha256": hashlib.sha256("\n".join(listing).encode()).hexdigest(),
                 "n_entries": len(listing), "n_entries_at_start": len(listing_before),
                 "listing_at_start_sha256": hashlib.sha256("\n".join(listing_before).encode()).hexdigest()}

def parse_metrics(metrics) -> dict:
    """Pure: sum counters by name from LLM.get_metrics() objects (name, value attributes; labels ignored)."""
    out = {}
    for m in metrics or []:
        name, val = getattr(m, "name", None), getattr(m, "value", None)
        if name is None or not isinstance(val, (int, float)):
            continue
        out[name] = out.get(name, 0) + val
    return out


def _scheduler_stats(llm):
    """Engine stats via LLM.get_metrics() (needs disable_log_stats=False). Preemptions are the scheduler-match
    criterion: with identical fixed caps (max_num_seqs, max_num_batched_tokens, num_gpu_blocks_override) and equal
    preemption counts the offline scheduler took the same path; max concurrent sequences is not exposed at this pin."""
    try:
        fn = getattr(llm, "get_metrics", None)
        m = parse_metrics(fn()) if callable(fn) else {}
        # exact metric names (Deviation 4 #4): a prefix match would hit vllm:prompt_tokens_cached
        return {"available": bool(m), "n_preemptions": m.get("vllm:num_preemptions"),
                "prefix_cache_hits": m.get("vllm:prefix_cache_hits"),
                "prompt_tokens": m.get("vllm:prompt_tokens"),
                "generation_tokens": m.get("vllm:generation_tokens"),
                "max_concurrent_running": None,
                "note": "max concurrent not exposed at this pin; match = equal caps + equal preemptions"}
    except Exception as ex:
        return {"available": False, "error": str(ex), "max_concurrent_running": None, "n_preemptions": None}


def _write_aborted(out_dir: Path, partial: dict, reason) -> dict:
    meta = dict(partial or {})
    meta["aborted"] = str(reason)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

def run(args) -> dict:
    for k, v in env_for_mode(args.mode, args.attention_backend).items():
        os.environ[k] = v
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    import platform as _platform
    partial_meta = {"run_id": args.run_id, "model": args.model, "mode": args.mode,
                    "image_digest": os.environ.get("R1_IMAGE_DIGEST", ""),
                    "platform_arch": _platform.machine(), "arch": _platform.machine(),
                    "declared_cuda_graph_mode": declared_cuda_graph_mode(args.mode)}
    cache_env, cache_info = _cache_setup(args.cache, args.cache_dir, out_dir)
    partial_meta["cache"] = cache_info
    try:
        assert_warm_cache_nonempty(args.cache, cache_info["n_entries_at_start"],
                                   is_warmup=bool(getattr(args, "warmup", False)))
    except Exception as e:
        _write_aborted(out_dir, {**partial_meta, "cache": cache_info}, e)
        raise
    os.environ.update(cache_env)
    import torch  # container-only; lazy relative to module import
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from vllm import LLM, SamplingParams  # lazy: vllm not installed on host/test venv
    items = [json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()]
    if getattr(args, "ids_file", ""):
        items = filter_items_by_ids(items, Path(args.ids_file).read_text().split())
        if not items:
            _write_aborted(out_dir, partial_meta, f"ids-file {args.ids_file} matched no items in {args.items}")
            raise RuntimeError(f"ids-file {args.ids_file} matched no items in {args.items}")
    caps = json.loads(args.caps_json) if args.caps_json else None
    # harness.generate always sends chat_template_kwargs={"enable_thinking": thinking}
    # (default False: Qwen3-1.7B paper-2 non-thinking; Qwen3-4B-Instruct-2507 has no thinking mode)
    ctk = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {"enable_thinking": False}
    canonical_ids = [it["id"] for it in items]
    submit_ids = permuted_order(canonical_ids, args.order_seed)
    by_id = {it["id"]: it for it in items}
    engine_kwargs = build_engine_kwargs(args.model, args.mode, args.num_gpu_blocks,
                                        args.attention_backend, args.max_num_seqs, args.max_num_batched_tokens,
                                        args.max_model_len)
    llm = LLM(**engine_kwargs)
    # Scheduler identity (B2): read back the resolved config; abort (run_meta with
    # "aborted", non-zero exit via the raise) if any requested value differs.
    expected = expected_engine_config(num_gpu_blocks=args.num_gpu_blocks,
                                      attention_backend=args.attention_backend,
                                      max_model_len=args.max_model_len,
                                      max_num_seqs=args.max_num_seqs,
                                      max_num_batched_tokens=args.max_num_batched_tokens,
                                      mode=args.mode)
    try:
        actual_config = read_engine_config(llm)
        mismatches = verify_engine_config(actual_config, expected)
        tf32_readback = {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                         "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}
        if tf32_readback["matmul_allow_tf32"] or tf32_readback["cudnn_allow_tf32"]:
            mismatches.append(f"tf32: requested off, got {tf32_readback}")
    except Exception as e:
        _write_aborted(out_dir, {**partial_meta, "requested_engine_config": expected}, e)
        raise
    if mismatches:
        reason = "engine config read-back mismatch: " + "; ".join(mismatches)
        _write_aborted(out_dir, {**partial_meta, "requested_engine_config": expected,
                                 "engine_config_readback": actual_config, "tf32": tf32_readback}, reason)
        raise RuntimeError(reason)
    tokenizer = llm.get_tokenizer()
    log_path = out_dir / "engine.log"
    t0 = time.time()
    all_records = {}
    for pass_n in range(1, args.passes + 1):
        prompts, meta = [], []
        for iid in submit_ids:
            it = by_id[iid]
            bench = it.get("benchmark")
            prompt = tokenizer.apply_chat_template(it["messages"], tokenize=False,
                                                   add_generation_prompt=True, **ctk)
            prompts.append(prompt)
            meta.append((it, bench))
        sps = [SamplingParams(temperature=0, max_tokens=max_tokens_for(b, args.max_tokens, caps),
                              logprobs=(args.logprobs or None), seed=0) for _, b in meta]
        outputs = llm.generate(prompts, sps)
        recs = []
        for (it, bench), out in zip(meta, outputs):
            o = out.outputs[0]
            rec = {"id": it["id"], "benchmark": bench, "text": o.text, "output": o.text,
                   "token_ids": list(o.token_ids), "finish_reason": o.finish_reason,
                   "n_out_tokens": len(o.token_ids),
                   "n_prompt_tokens": len(out.prompt_token_ids) if out.prompt_token_ids else None}
            if args.logprobs and o.logprobs:
                rec["first_logprobs"] = [{str(k): {"logprob": v.logprob, "decoded_token": v.decoded_token}
                                          for k, v in lp.items()} for lp in o.logprobs[:1]]
            recs.append(rec)
        recs = canonical_order(recs, canonical_ids)
        dest_root = out_dir / f"pass{pass_n}" if args.passes > 1 else out_dir
        for (domain, bench), group in group_items(recs).items():
            p = dest_root / "gen" / domain / f"{bench}.dev.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in group), encoding="utf-8")
        all_records[f"pass{pass_n}"] = len(recs)
    wall = time.time() - t0
    scheduler_stats = _scheduler_stats(llm)
    try:
        assert_prefix_cache_zero(scheduler_stats)  # B3: prefix-cache hits must be 0
    except Exception as e:
        _write_aborted(out_dir, {**partial_meta, "scheduler_stats": scheduler_stats,
                                 "engine_config_readback": actual_config, "tf32": tf32_readback}, e)
        raise
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    try:
        log_info = assert_engine_log_ok(log_text, mode=args.mode,
                                        attention_backend=args.attention_backend)
    except Exception as e:
        _write_aborted(out_dir, {**partial_meta, "scheduler_stats": scheduler_stats,
                                 "engine_config_readback": actual_config, "tf32": tf32_readback}, e)
        raise
    try:
        cfg = str(llm.llm_engine.vllm_config)
    except Exception as e:
        cfg = f"unavailable: {e}"
    run_meta = {"run_id": args.run_id, "model": args.model, "mode": args.mode, "passes": args.passes,
                "ids_file": getattr(args, "ids_file", ""), "warmup": bool(getattr(args, "warmup", False)),
                "order_seed": args.order_seed, "max_tokens": args.max_tokens, "caps": caps,
                "logprobs": args.logprobs, "chat_template_kwargs": ctk, "items": str(args.items),
                "n_items": len(items), "records_per_pass": all_records, "wall_time_s": round(wall, 3),
                "engine_config": cfg, "engine_config_readback": actual_config,
                "requested_engine_config": expected, "env_fingerprint": env_fingerprint(),
                "image_digest": os.environ.get("R1_IMAGE_DIGEST", ""),
                "platform_arch": _platform.machine(), "arch": _platform.machine(),
                "declared_cuda_graph_mode": declared_cuda_graph_mode(args.mode),
                "expected_cudagraph_pair": list(expected_cudagraph_pair(args.mode)),
                "cudagraph_mode_readback": actual_config.get("cudagraph_mode"),
                "breakable_cudagraph_readback": actual_config.get("breakable_cudagraph"),
                "env_vars_set": {**env_for_mode(args.mode, args.attention_backend), **cache_env},
                "attention_backend_selected": log_info["attention_backend"],
                "cuda_graph_capture_state": log_info["cuda_graph_capture_state"],
                "cuda_graph_mode": actual_config.get("cudagraph_mode"),
                "tuned_config": log_info, "scheduler_stats": scheduler_stats, "cache": cache_info,
                "tf32": tf32_readback,
                "note_scheduler_stats": "preemptions reported only, never a discard criterion; prefix-cache hits must be 0"}
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    return run_meta

def build_parser():
    ap = argparse.ArgumentParser(prog="r1_offline.py")
    ap.add_argument("--model", required=True); ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--run-id", required=True)
    ap.add_argument("--mode", choices=["bi", "default"], default="default")
    ap.add_argument("--passes", type=int, choices=[1, 2], default=1)
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--caps-json", default="")
    ap.add_argument("--logprobs", type=int, default=0)
    ap.add_argument("--num-gpu-blocks", type=int)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=8192)  # paper 2 serving: prompt + 4,096 new tokens
    ap.add_argument("--attention-backend")
    ap.add_argument("--cache", choices=["cold", "warm"], default="warm")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--ids-file", default="", help="keep only these item ids from --items (file order kept)")
    ap.add_argument("--warmup", action="store_true", help="warmup run: fills the warm cache, empty start allowed")
    ap.add_argument("--chat-template-kwargs", default="")
    return ap

def main(argv=None):
    return run(build_parser().parse_args(argv))

if __name__ == "__main__":
    main()
