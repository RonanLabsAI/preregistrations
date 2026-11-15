# Generation result

Operational stop at the counter-profile validity gate; no code candidate ran and no audited scientific null is claimed. Source: `receipts/results.jsonl`.

| Matching champion metric | Mean tok/s | CI | Denominator |
|---|---:|---|---|
| C1 native reply decode | 81.785376 | [81.783530, 81.787221] | n=5 rounds |
| C16 aggregate client | 272.577415 | [272.510761, 272.644068] | n=5 rounds |

Both matching controls pass their frozen bands; all exact output IDs match, n=192 replies including excluded warm rounds. Metrics retain separate denominators.

| Timeline cell | Native overhead | Paired ratio CI | Denominator |
|---|---:|---|---|
| c1 | 0.082843% | [1.000590, 1.001067] | n=5 pairs |
| c16 | 0.247041% | [1.001854, 1.003087] | n=5 pairs |
| 8192 | 0.014653% | [0.999940, 1.000353] | n=5 pairs |
| 32768 | 0.012217% | [1.000056, 1.000189] | n=5 pairs |

The counter-instrumented reply took 5.850585x native time, or 485.059% increase, n=1 profiled reply versus n=5 controls; no ratio CI is available. This exceeds the frozen validity cap and halts code work.

Replay-only achieved occupancy median 24.10%, range [23.88, 24.20], n=5 kernel instances. These intrusive counters are diagnostic only and cannot set a candidate bound.

Independent MMA: 602.051373 TFLOP/s, CI [601.973221, 602.129525], n=5 samples. These are sustained references, not certified absolute ceilings.

Independent STREAM: 4858.222990 GB/s, CI [4856.516838, 4859.929142], n=5 samples. These are sustained references, not certified absolute ceilings.

Host readback: unrun, bound inconclusive. EXL3 strips: unrun behind counter-overhead gate. Guard widening: unrun, no eligible remaining path. Candidate speed ratios and intervals are N/A; no falsifier is counted.

Final daily restore: 113.972s to health and 115.431s to both routed answers, n=1 manual cycle. Earlier smoke restore is separately retained.

LEFT exclusive reservation cost: 1.415772 GPU-hours, n=2 reservation windows, including waits and restore. Shared resident fallback GPU busy time was not metered. All claims are released.

Next: a separately preregistered low-overhead counter/critical-path method, followed by prospective bounds before any code candidate. Production service is restored. No post, paper or upstream submission was made.
