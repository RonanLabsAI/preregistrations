# Verdict: Ollama's DGX Spark gpt-oss headline on our stack

Preregistration: https://github.com/RonanLabsAI/preregistrations/blob/e21101fa15381704d40972fdffa9d7ab4f01b823/studies/vendor-headline-check/amendment-a1-2026-11-14.md

Decode: **consistent with claim** under the preregistered equivalence band.
Median client-side decode rate was 42.01 tokens/s (order-statistic 95.9% CI [41.98, 42.10], n=20 runs) against the vendor's 41.14 tokens/s, from receipts/results.jsonl.
Prefill: **above claim** under the preregistered equivalence band.
Median overhead-differenced prefill rate was 1746 tokens/s (order-statistic 95.9% CI [1693, 1778], n=20 paired runs) against the vendor's 1169 tokens/s, from receipts/results.jsonl.
Valid paired trials were 20/20 attempted trials, from receipts/results.jsonl.

## What this does and does not establish

It tests the vendor's single-stream numbers for the same model weights and workload on the same hardware class, as served by our current production engine version and configuration, measured at the client through the router. Client-side decode timing includes streaming transport; it can only understate the engine's own rate. It does not test the vendor's older engine version, driver or an unloaded-model setup, and it does not measure multi-user throughput. The environment fingerprint is recorded in manifest.json.
