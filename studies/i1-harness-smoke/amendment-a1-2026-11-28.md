# I1 harness smoke v1, amendment A1 (2026-11-28 PT): rerun after harness fix

Reason: unit `sparksmoke-1` failed P4 and only partly met P1 because of two harness defects (see `outcome-2026-11-28.md`). This amendment changes no inputs, arms, criteria or analysis.

Changes:

1. A new unit `sparksmoke-2` (arm `spark`, seed 2) reruns the same procedure, inputs, job and pass criteria P1 to P5 on the same DGX Spark.
2. The fixed harness copies frozen inputs into the package, records each start in `receipts/attempts.log`, passes the attempt count to the job as `IDLEWORKS_ATTEMPT`, and fails a run that produced no results.
3. The environment fingerprint changes only in its harness component: `gb10|drv580.142|vllm0.30.1rc1.dev245@3af6a45b060d|qwen3-0.6b@f47f71177f32|ctx4096|gmu0.12|eager|job@4e12bb6a18f4|harness@63c3ddf`, where `63c3ddf` is the harness fix commit.

The outcome of `sparksmoke-1` stands as recorded.
