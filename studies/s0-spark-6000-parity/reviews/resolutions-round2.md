Resolutions of review round 2:

Open round-1 points:
2. Backend from serve.log is now a unit check: serve.log must exist and show exactly ["FLASH_ATTN"].
4. Every attempt that produced records must have a treatment record; conflicting duplicates are now a unit problem.
5. Driver and compute capability are checked exactly against the card (580.142 / 12.1 Spark; 595.99.02 / 12.0 RTX PRO 6000).
6. Batch and control comparisons are INCOMPLETE unless the lower-concurrency unit's peak running requests are below the c128 unit's; a running-request histogram is reported per unit.
9. A missing GPU UUID is a treatment problem; the same-card check requires every unit to report exactly one UUID and all to match.
10. Costed unit by unit in the card; caps raised to 5.0 and 5.0 GPU-hours.
13. With the new Spark batch arm, HARDWARE-DIFFERENT now means both devices pass the same within-device checks (replicate and two concurrencies each); wording updated.

New points:
B1. Verdict order is now INCOMPLETE, NONDETERMINISTIC, BATCH-VARIANT, HARDWARE-DIFFERENT, UNINFORMATIVE, PASS. UNINFORMATIVE can only fire when the five core comparisons are all IDENTICAL.
B2. Added m-spark-c64 (concurrency 64; 32 would cost too much on GB10) and batch_spark; required IDENTICAL before HARDWARE-DIFFERENT or PASS.
B3. For two fixed-pool units, differing logged KV cache sizes make the comparison INCOMPLETE; a missing serve.log is a unit problem.
B4. The v1_gmu085 flag is removed; kv_pool_6000 is reported only and licenses nothing (card says so).
B5. The VLLM_* environment must equal the image's exact list (base six, plus VLLM_BATCH_INVARIANT=1 for BI units).
B6. Per-benchmark paired counts, both scores and their difference per comparison; reference files verified against their frozen SHA-256 and skipped if different; unreadable score or reference files are noted and skipped, verdicts still written.
B7. The job applies the analysis schema before writing a record; a failing reply is counted as a failure (exit 1) and retried on resume.
