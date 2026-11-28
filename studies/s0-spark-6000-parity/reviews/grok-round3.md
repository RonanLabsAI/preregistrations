Seven round-2 points are closed. Three new blockers would force every model `INCOMPLETE` or would pool bytes the verdict never checked.

**(A)**

2. **RESOLVED.** A missing `serve.log`, an attention-backend list other than `["FLASH_ATTN"]`, a KV-size list whose length is not 1, or no running-request lines is appended to the unit's `problems`, so that unit's comparisons are `INCOMPLETE`.

4. **RESOLVED.** Conflicting duplicate outputs are added to `problems`, and any result `attempt` with no treatment record is added to `problems`.

5. **RESOLVED.** `check_treatment` requires Spark `580.142` / `12.1` and RTX PRO 6000 `595.99.02` / `12.0`, plus the card's GPU name strings.

6. **RESOLVED.** Each unit stores `max_running_reqs` and `running_reqs_histogram_16`, and `batch_spark`, `batch_6000`, and `control_bi_off` are `INCOMPLETE` unless the lower-concurrency peak is strictly below the other peak.

9. **RESOLVED.** A missing UUID is a treatment problem, and `same_card` requires every unit of that model and hardware class to be present with exactly one UUID, all equal.

10. **RESOLVED.** The stated per-unit minutes sum to 184 + 54 on the Spark (9 units, about 4.0 GPU-hours) and 144 + 72 on the RTX PRO 6000 (18 units, about 3.6), both under the 5.0 caps.

13. **RESOLVED.** `HARDWARE-DIFFERENT` is reached only after both replicates and both batch arms are `IDENTICAL`, and the card's wording matches that.

B1. **RESOLVED.** `UNINFORMATIVE` is after `NONDETERMINISTIC`, `BATCH-VARIANT`, and `HARDWARE-DIFFERENT`, so it runs only when all five core comparisons are `IDENTICAL` and the control is `IDENTICAL`.

B2. **RESOLVED.** `m-spark-c64` and `batch_spark` are in the unit table and in `components()`, and both `HARDWARE-DIFFERENT` and `PASS` require that arm to be `IDENTICAL`.

B3. **RESOLVED.** A missing `serve.log` or a backend list other than `["FLASH_ATTN"]` fails the unit, and two `kv16g` units with different `kv_cache_tokens` make that comparison `INCOMPLETE`.

B4. **RESOLVED.** There is no `v1_gmu085` flag; `kv_pool_6000` is only under `reported`, and `pooling_allowed` is true solely when the verdict is `PASS`.

B5. **RESOLVED.** Each unit's sorted `VLLM_*` list must equal the frozen six base variables, plus `VLLM_BATCH_INVARIANT=1` on batch-invariant units, so units in one comparison share that list.

B6. **RESOLVED.** Comparisons emit per-benchmark `both` / `neither` / `a_only` / `b_only`, both scores, and `score_diff_b_minus_a`; reference SHA-256s are checked; a bad score or reference file is recorded in `errors` and `analysis.json` is still written.

B7. **RESOLVED.** `s0_job.py` raises before append when output, finish reason, or token counts fail the analyzer schema, increments `n_fail`, returns 1, and `read_done` will request that id again.

**(B)**

1. **Blocker.** Card: `vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063`. Code: `BASE_IMAGE` is that digest, and `t["image_id"] = ins.get("Image")` is compared with it for every non-BI unit. `.Image` is the local image ID; the card's `@sha256` form is the registry manifest digest (the BI arm correctly uses a separate local ID, `sha256:c92aec79...`). Every control unit fails the image check, `control_bi_off` is `INCOMPLETE`, and rule 1 makes every model `INCOMPLETE`. **Fix:** for the base image, require a `RepoDigests` entry of `vllm/vllm-openai@sha256:3af6a45b...`; keep `.Image == sha256:c92aec79...` for BI units only, and record the base image's local ID only if you also check that field.

2. **Blocker.** `t["args"] = ins.get("Args")` and `strip_model_and_port(...) != want_args`, with `want_args` starting `--served-model-name`, then `COMMON`, then `MEM`. Docker's top-level `Args` is the process argv after `Path`; the vLLM image entrypoint puts `serve` first, and the card never places `--served-model-name` immediately before `--dtype`. Exact equality fails on every unit, so every verdict is `INCOMPLETE`. **Fix:** drop a leading `serve` (and any `-m` / module path), then require the frozen `--flag` / value pairs to be present, with `--served-model-name` equal to the served model, and reject any other `--flag`.

3. **Blocker.** Card: "For Qwen3-1.7B, a PASS adds the RTX PRO 6000 to our earlier result" and, later, "A match or a mismatch there decides nothing." A `PASS` only means the new Spark run matches the new 6000 run. The October Spark/GB300 files can disagree and still leave `pooling_allowed` true, so the 6000 bytes would be mixed into a corpus they do not match. **Fix:** state that `PASS` licenses pooling only of these new Spark and RTX PRO 6000 runs; add an October pool flag that is true only when all three reference comparisons are 1,179/1,179.

4. **Should-fix.** Card: "hardware at the other batch: m-spark-c64 vs m-6000-c32" and `reported()` uses that pair. Those arms are client concurrency 64 and 32. The primary hardware arm (`m-spark-c128` vs `m-6000-c128`) also never requires the two running-request histograms to overlap, so both devices can be invariant inside disjoint server-batch ranges and still yield `PASS` or `HARDWARE-DIFFERENT`. **Fix:** if the two c128 histograms are disjoint, set `hardware` to `INCOMPLETE`; relabel the 64-vs-32 pair as unequal concurrency and keep it out of hardware wording.

5. **Should-fix.** `re.findall(r"KV cache size: ([\d,]+) tokens", text)` then `len(log["kv_cache_tokens"]) != 1` fails the unit. That pattern matches both `GPU KV cache size` and `CPU KV cache size`. A second line makes every fixed-pool unit fail. **Fix:** parse only `GPU KV cache size: ([\d,]+) tokens` and require that one value.

6. **Should-fix.** Card: "maximum lost work 5 minutes (items still in flight)". `s0_job.py` defaults `--timeout` to `900.0`. One in-flight request can run 15 minutes, and `--concurrency` of them can be in flight together. **Fix:** set the contract's lost-work bound to the timeout, or lower `--timeout` to 300 seconds.

Points remain open: none of the round-2 list; the three new blockers remain open.
