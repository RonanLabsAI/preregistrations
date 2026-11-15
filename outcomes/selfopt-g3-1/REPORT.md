# SelfOptimize G3.1 outcome

Recorded 2026-11-14T21:58:34.599366-08:00 (PT).

Source: `receipts/results.jsonl`; bound sensitivity: `receipts/host-bound-sensitivity.json`.

A valid timeline is retained independently of the counter gate. Candidate outcomes below are gate dispositions; no unrun candidate is counted as a scientific falsifier.

| Champion remeasurement | Mean tok/s | t95 CI (n=5 rounds) |
|---|---:|---|
| C1 | 81.783164 | [81.780889, 81.785439] |
| C16 | 272.650814 | [272.589889, 272.711740] |

| Instrumentation | Native overhead | Client overhead | Decision |
|---|---:|---:|---|
| C1 timeline | 0.4511% | 0.4528% | PASS 5% |
| C16 timeline | 0.1641% | 0.1625% | PASS 5% |
| One sampled counter reply | 602.5319% | 591.8434% | REJECT 10% |

Counter comparison is one profiled reply versus five native controls: no ratio CI. Timeline ordered-ratio intervals and both inactive/native comparator samples are in summary.json. Neither overhead comparison is a randomized causal estimate.

| Decode timeline (medians across 5 reports) | C1 first prompt | C16 stable full-width rounds |
|---|---:|---:|
| Device-idle upper fraction (%) | 2.9173 | 2.6791 |
| Free-idle speed ceiling | 1.0300 | 1.0275 |
| EXL3 union / decode (%) | 36.0485 | 59.0868 |
| Dense BF16 union / decode (%) | 31.7347 | 7.6385 |

Family unions can overlap; they do not form an additive time budget. All cycle definitions, stream overlap, top kernels, API synchronization costs and per-report values are retained in summary.json. A blocking host synchronization mostly waiting on active device work is not removable host time.

| Candidate | Bound | Speed interval | Disposition |
|---|---|---|---|
| host-readback | C1 1.030183x; C16 1.028311x (maximum observed) | N/A (unrun) | Even deleting all within-decode device-idle gaps falls below 1.05; no demonstrated independent C1 host work. No patch/run. |
| exl3-strips | Unknown | N/A (unrun) | EXL3 time is measured, but counter evidence failed its cap and no applicable decoder ALU ceiling exists. MMA/STREAM and occupancy are not that ceiling. No patch/run. |
| one-guard | 1.000000x | N/A (unrun) | No eligible closed sm103 guard: qmmf admits sm103; native pair decode is not this EXL3 path; AHEAD remains excluded. No patch/run. |

C1 profiling covers the preregistered first prompt; C16 covers the frozen sixteen-prompt round. Candidate bounds apply to those measured paths.

No candidate promoted; the champion remains the frozen engine. No audited scientific null is claimed.

LEFT exclusive reservation: 0.554575 GPU-hours, including load, CPU wait, profiling and restore. Shared resident fallback GPU busy time was not metered.
Preserved daily restore: 101.697s to health, 103.121s to both routed LEFT answers. Claims released; production router and RIGHT identities unchanged.

Measured implementation source heads (later report-only commits are supplied at handoff):
- SelfOptimize: `feat/w26-g3-1` at `602daefcc0ce2d2b0fe09c5530e8f5c30a938cdf`.
- TensorFold_engine: `ronanlabs/sm103-enable` at `7c1bc0be3df413f43c65fb278699b629a44e7a90`.

Only the preregistration repository was pushed. Raw profiler/database output is retained privately by hash; no private operational receipts are included here.

Next: use the accepted timeline to preregister an applicable EXL3 decoder throughput experiment, or a concrete finite bound for another measured mechanism, before any patch.

Conservative post-analysis host sensitivity (deliberately double counting all DtoH API time and adding the positive overhead interval upper estimate) remains below the bar: C1 1.037719x, C16 1.034469x; this is a robustness check, not a changed admission rule.

This is a PI-authorized manual implementation run with local durable checkpoints. No new Mission Control product was queued/admitted, and no MC interruption/heartbeat guarantee is claimed.

Synchronization attribution: C1 median host wait 4.272653s includes only 0.110876s without device activity; C16 median wait 7.280956s includes 0.176519s without device activity (n=5 reports per cell). A long wait is predominantly dependent GPU execution.
