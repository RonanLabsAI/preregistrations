The matched-KV primary arms, split verdict labels, treatment checks, and per-model control are in the card and the code. Two new verdict bugs remain: a matching control relabels real differences `UNINFORMATIVE`, and Spark batch dependence is never measured. Seven round-1 points are only partly closed.

**(A)**

1. **RESOLVED.** Primary arms on both chips use `--gpu-memory-utilization 0.25` and `--kv-cache-memory-bytes 17179869184`; `0.85` with no fixed pool is only `m-6000-gmu085-c128` and stays out of the verdict. Declining `--enforce-eager` and prefix-cache off holds: the pooling license is for the config V1/R1 serve, and the new same-process replicates cover within-device graph and prefix noise. Exact on Paper did not vary hardware, so it does not show those defaults are safe across chips; that is what this study measures.

2. **PARTLY.** `FLASH_ATTN` is in `COMMON` and `check_treatment` rejects any other argv. The backend parsed from `serve.log` is stored on `server_log` and never added to `problems`, so a logged backend mismatch can still be `PASS` or `HARDWARE-DIFFERENT`. No decline was given for that half.

3. **RESOLVED.** Hardware, per-device determinism, `batch_6000`, and the control are separate comparisons, and a batch or replicate mismatch is labeled before hardware. Mismatches carry `attempt_a` / `attempt_b` and `by_attempt`. Not gating on unequal attempt sets is the right omission: a resume is a valid run, and the same-config replicate is what separates a glitch from a chip effect.

4. **PARTLY.** Off-protocol treatment files fail the unit, exact duplicate lines no longer force `INCOMPLETE`, and `results_sha256` is recorded. A result’s `attempt` is never required to have a `treatment-a<N>.json`, and `conflicting_duplicates` is not added to `problems`, so the kept row can still `PASS` while the file holds a second output.

5. **PARTLY.** The card states both hosts are aarch64, `machine == "aarch64"` is required, and each arm’s image id is checked (`BI_IMAGE` vs `BASE_IMAGE`). `gpu.driver` and `compute_cap` are stored and never compared to 580.142 / 595.99.02 or 12.1 / 12.0. Treating the unequal drivers as part of the platform, rather than as a third verdict, holds: they cannot be equalized, and the license is for these two platforms. That does not justify accepting a driver other than the one frozen in the card.

6. **PARTLY.** Every model has `nobi-c128` vs `nobi-c32`; a missing control makes the comparison `INCOMPLETE`; a real string mismatch is `DIFFERENT` and is not itself a failure; a fully identical control yields `UNINFORMATIVE` with `pooling_allowed` false. `server_log_facts` keeps only `max_running_reqs`, so the card’s running-request counts are not in the receipt and nothing requires c128 and c32 to have seen different batches.

7. **RESOLVED.** The four-unit sentence is gone. `expected_units()` has eight units per model, and `PASS` is the fall-through after the five comparisons.

8. **RESOLVED.** Qwen’s `config.json`, `generation_config.json`, and `tokenizer_config.json` are hashed; Granite’s `config.json` plus the shared `generation_config.json`, `tokenizer_config.json`, and `chat_template.jinja` are hashed. Any `n_prompt_tokens` disagreement makes that comparison `INCOMPLETE`.

9. **PARTLY.** The 6000 name is matched exactly, `same_card` unions UUIDs within a hardware class, and the procedure keeps the 84 GiB start gate. `len(cards) <= 1` is true when every UUID is missing, and a missing UUID is not a treatment problem, so the one-card rule can pass with no card id. Declining a runtime free-memory floor on the 6000 holds: the process reserves the pool at start on a worker-only card, and a later process cannot shrink that reservation.

10. **PARTLY.** A cap hit leaves finished models judgeable, and `analyze.py` does that. The 3.1 h / 4.0 h RTX figures are not a sum over the 18 units. After the stated 4 minutes of cold start, 1.9 h remains for 18 runs, about 6 minutes of compute each, while the only measured run is 408 s and six units are specified at twice the concurrency-128 wall time.

11. **RESOLVED.** The procedure runs `score.py` once per unit, then one `analyze.py` with `--scores`. Paired counts are produced only when those files are passed.

12. **RESOLVED.** Missing content is stored as JSON null. A kept row must have a string output, `finish_reason` in `{stop, length}`, and integer token counts. `first_diff` runs only after that string check.

13. **PARTLY.** PASS is worded per model, reference comparisons are written only to `out["reference"]`, the Aya result is labeled an F1 shift, the kernel files and functions are named, and Granite 4.1 is stated as dense `GraniteForCausalLM`. `HARDWARE-DIFFERENT` still says both devices passed a within-device batch-invariance check. Spark has no batch arm.

14. **RESOLVED.** Both scripts reject a bad prompt hash or a count other than 1,179. After any unit finishes, an amendment cannot change units, flags, comparisons, checks, or verdict rules. Unparseable lines are skipped, and the job starts the next append on a fresh line. Declining a non-zero exit on a scientific failure holds: prompt and I/O failures already exit non-zero, `main` returns 0 only after writing `analysis.json`, and the procedure reads the verdict from that file.

**(B)**

1. **Blocker.** Card: “2. UNINFORMATIVE if control_bi_off is IDENTICAL.” Then rules 3–5. Code: `elif ctl["status"] == "IDENTICAL": verdict = "UNINFORMATIVE"` sits above the `NONDETERMINISTIC`, `BATCH-VARIANT`, and `HARDWARE-DIFFERENT` branches. A control that matches relabels an observed hardware, batch, or replicate difference. Pooling stays off, but R1 is not told to treat hardware as a factor and V1 is not told to score that model on the 6000 only. **Fix:** apply `UNINFORMATIVE` only when the control is `IDENTICAL` and the other four comparisons are `IDENTICAL`. If any of those four is `DIFFERENT`, let rules 3–5 fire.

2. **Blocker.** Card: “each device must reproduce itself, and its outputs must not depend on batch composition. Both are measured here too.” And: “two devices that each pass vLLM’s within-device batch-invariance check.” `components()` has only `batch_6000` (`m-6000-c128` vs `m-6000-c32`). Spark units are c128 and c128-r2. The chips run at different speeds, so the same client concurrency is a different server batch. A Spark-only batch effect becomes `HARDWARE-DIFFERENT`, and a `PASS` licenses Spark outputs at batch shapes that were never run. **Fix:** add `m-spark-c32` and `batch_spark`, and require it `IDENTICAL` before `HARDWARE-DIFFERENT` or `PASS`. Otherwise limit the pooling sentence to client concurrency 128 and drop the claim that Spark batch independence was measured.

3. **Should-fix.** Card: “the scheduler has the same number of KV blocks on each.” `server_log_facts` stores `kv_cache_tokens` and, if `serve.log` is missing, `{"serve_log": "absent"}`. `complete` never reads either, and the two sides are never compared. A hardware verdict can rest on unequal or unknown pools. **Fix:** if the log is missing, the backend list is not exactly `["FLASH_ATTN"]`, or the KV token counts differ across the units of a comparison, that comparison is `INCOMPLETE`.

4. **Should-fix.** Card: `v1_gmu085_runs_poolable` says whether V1’s 0.85, no-fixed-pool outputs can be pooled too. Code: `verdict == "PASS" and kv == "IDENTICAL"`, with a single `m-6000-gmu085-c128` and no replicate in `expected_units()`. One unreplicated match licenses that regime. **Fix:** report `kv_pool_6000` and leave the flag false, or add a gmu085 replicate and set the flag only when that replicate matches too.

5. **Should-fix.** `check_treatment` only tests `("VLLM_BATCH_INVARIANT=1" in env) != bi`. The job records every `VLLM_*` variable, and any other variable can differ between hosts without failing the unit. **Fix:** require the bi env list to be exactly `["VLLM_BATCH_INVARIANT=1"]` and the nobi list to be empty, and require the lists to match across units that share a comparison.

6. **Should-fix.** Card: “For each comparison, we report the paired counts … and the score difference per benchmark.” `compare` writes one aggregate `correct_pairs` and no score difference. `score.py` writes per-benchmark scores to `*.bench.json`, which `analyze.py` never reads. `json.load` of `--scores`, and `load_ref`’s `r["id"]`, run before `analysis.json` is written, so a bad score or reference file raises and the verdict file is never produced. Reference hashes in the card are not checked. **Fix:** load bench scores, emit paired counts and the score difference per benchmark, verify the reference hashes, and on any failure record it and still write the verdicts.

7. **Should-fix.** `s0_job.py` appends `"output": (ch.get("message") or {}).get("content")` together with whatever `finish_reason` and token counts came back, then `read_done` treats that id as done. `n_fail` increments only in the `except` path, so a 200 with null content, a bad finish reason, or null token counts exits 0 and is never resumed. The model stays `INCOMPLETE` until someone hand-edits the file. **Fix:** if the record fails the same schema `analyze.py` enforces, do not append it, count a failure, and return 1 so the existing resume retries the item.

Points 2, 4, 5, 6, 9, 10, and 13 remain open, and the revised card adds two blockers.
