The confirmatory script is the piece that can still lock the wrong test. I am checking the vLLM pin's block-size and metrics behavior so the scheduler verdict is exact.

Two open blockers. Deviations 1 and 2 are valid pre-outcome corrections. The offline runner sets the locked knobs. The confirmatory script and the scheduler-identity check do not.

### Round 1

**Blockers**

1. Scheduler mix at concurrency 96 — **partly**. Primary arm is offline `LLM.generate`, one order, shared `max_num_seqs` 32, `max_num_batched_tokens` 8192, prefix caching off, `num_gpu_blocks_override` 2048. Deviation 2 replaces realized max concurrency with equal preemption counts. Block size is still platform-chosen, and the check is not enforced (blocker B).

2. One image is not one binary — **partly**. Deviation 1 is right: both Stations are Grace, stratum S is the arm64 image digest, confirmatory pair unchanged. The launcher forces `FLASH_ATTN`. `run_meta` has no digest, `launch.py` hardcodes GB10 as stratum G, and a backend or graph mismatch is never discarded.

3. Rane arm — **resolved** in the card: offline Qwen3-4B, Rane output caps, 3 warm restarts, RTX 5080, RTX PRO 6000, and GB300, plus an item mismatch list if Zenodo items are absent.

4. H1 multiplicity — **partly**. Prose is one endpoint and one pair. The script pools other runs (blocker A).

5. H1 rejection region — **resolved** in prose: excess > 0 with the 95% interval above 0; one mismatched item fails that H1b pair; failures are not excused. The script emits no such verdict.

**Majors.** Four devices, not chip classes — **resolved**. Cold/warm split — **partly**: the card separates cold-cold, cold-warm, warm-warm, same-process, and order (two colds); the script's `warm-warm` includes order and same-process, and `load_run` keeps one pass. Factor screen, budget split, adapter control (same 2,244 items, interleaved-same vs mixed, one chip), cost rule (under 50% loss on both datacenter devices), primary logprobs off, McNemar as a direction check, set-size diagnostic, equal-benchmark weights, narrowed novelty — **resolved** in the card. The queue cannot launch adapter, HTTP, cost, or logprob arms; `limit=100` is not an `r1_offline.py` argument; zero-flip cluster bounds are absent. Public prereg — **not** (`preregUrl` pending).

### Blocker A — hashed analysis is a different test

`analyze_manifest` feeds H1a every warm `mode=default` transfer run: 3 warm restarts, 2 order permutations, and the same-process run. Order pairs inflate the floor and the cross rate and pull the excess toward 0. `load_run` overwrites `pass1` with `pass2`.

H1b pairs every `mode=bi` run with every other, so Qwen3-1.7B adapter, order, and full-set runs sit beside Qwen3-4B restarts. There is no pass/fail over the six device pairs or the within-device cold and warm restarts.

`h1a()` matches the card once the inputs are the three warm restarts: equal-benchmark mean, self-pairs dropped, one shared cluster bootstrap, rng 0, ratio only if the floor interval's upper bound is at least 0.5%, support only if the excess interval lies above 0. Deviation 2 (MMLU-Pro categories, other items singly, within benchmark) matches the code and overrides the card's "else benchmark" sentence.

Missing `scores.json` is stored as `correct=0`. Graded flips are then 0.

**Fix, Deviation 3, re-hash before admission.** H1a: `kind=restart`, `cache=warm`, Qwen3-4B, transfer, batch-invariant off, three runs a device. H1b: `kind=restart`, batch-invariant on, Qwen3-4B only; a pair fails if any item differs in text or `token_ids`. Emit the six cross-device pairs, the within-device restart pairs, and one overall pass. Abort if any confirmatory id is unscored. Keep both same-process passes for H2.

### Blocker B — caps plus preemption counts do not identify the schedule

Offline admission depends on prompt length, order, block size, block count, and the two batch caps. Speed does not enter while those match. `block_size` has no static default; the platform sets it, so the GB300 (sm100) and RTX PRO 6000 (sm120) can hold different token budgets at 2048 blocks.

`vllm:num_requests_running` is a gauge. A read after `generate` is the teardown value, so two zeros would match. `vllm:num_preemptions_total` matches across many schedules. After outputs diverge, a preemption gap is a consequence of the flips; discarding those runs drops the discrepant draws. Analysis never compares the counts. A missing metric stays `None` and would pass a hand check.

**Fix:** pin `block_size=16` and `kv_cache_dtype="bfloat16"` (`auto` is model dtype today). After `LLM()` init, read back block size, allocated blocks, and KV dtype; abort on a mismatch. Require equal prompt-token totals. Report preemption counts. Keep the run in H1a.

### Majors

**Stack gates.** TF32 is set before the vLLM import, then `run_meta` writes `false` as a literal. H1a is batch-invariant off, where vLLM does not force IEEE fp32. The engine log is this process's redirected stdout, read before exit, so the tuned-config check can pass on an empty file. Prefix-cache hits are unchecked (`vllm:prefix_cache_hits` exists). `R1_IMAGE` can replace the pin. Stratum cannot be decided from `run_meta`.

**Fix:** `NVIDIA_TF32_OVERRIDE=0`, `python -u`, and a readback after init of TF32, backend, prefix hits, graph mode, and image digest. Abort before scoring on a mismatch. Batch-invariant Blackwell runs on this pin use breakable CUDA graphs; declare graph mode per mode so a BI run is not discarded for differing from the non-BI capture. Same Station digest is a precondition for H1a.

**Rane.** The matched arm changes output caps only. List `max_num_seqs`, batched tokens, and the block override as pre-declared differences, or match his engine args on that arm only.

**Explain-away, closed by the same edits.** Hand discard on preemptions or backend after scoring. "Cold-warm pairs remain the H1b risk" and the sm10x uncertainty sentence: a mismatch is a failed pair, with no exception footnote. The 100-item smoke (within-device byte identity, both modes; devices not compared) did not move the rejection region. Name those item ids and do not cite them for H1b.

Merge the preregistration PR after Deviation 3 so the external timestamp is the corrected hash.

### What holds

Deviation 1 leaves H1a on one image. Deviation 2's cap (30 to 55 GPU-h) is a timing change. The runner sets `VLLM_BATCH_INVARIANT` and `VLLM_ATTENTION_BACKEND` before importing vLLM, prefix caching off, logprobs off unless asked, the block override, temperature 0, seed 0, a fresh cold directory per run, and a warm directory per device, model, and mode. Shuffled submits are written back in canonical order. The queue does not compare devices.
