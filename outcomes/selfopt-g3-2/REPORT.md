# W27 SelfOptimize G3.2 report

Recorded 2026-11-14T22:22:57.035534-08:00 (PT).

G3.2 produced isolated original-binary EXL3 timings and counters without unloading the daily. **The requested finite applicable candidate bound was not established. No candidate ran; no new champion.** This is a method gate result, not an audited scientific null.

Preregistration: https://github.com/RonanLabsAI/preregistrations/tree/1820646/protocols/selfopt-g3-2
Amendment before valid samples: https://github.com/RonanLabsAI/preregistrations/tree/31bce7e/protocols/selfopt-g3-2

## Harness validation

Exact champion fatbin and launch grids, MCG 4-bit, dimensions 4096/2048, eight slots, 256 experts, fixed splits and original resources. All six retained cases pass finite outputs, unique-count checks and byte-repeatability. Each stage has n=5 retained block medians of twenty replays after one warm block; intervals below are t95, df4. Served targets are medians of five G3.1 report medians. Intervals describe repeat precision, not transfer uncertainty.

C1 membership is equivalent under expert relabeling but weights and values are synthetic. C16 expert-ID/member arrays were not retained by Nsight Systems. Its distinct-expert sweep is synthetic and not exhaustive. Duration agreement cannot recover that provenance.

| Stage | C1 served us | C1 harness us [t95] | Ratio | Within 15% | C16 served us | Synthetic U96 us [t95] | Ratio | Within 15% |
|---|---:|---|---:|---|---:|---|---:|---|
| group | 3.904 | 4.8416 [4.8084, 4.8748] | 1.2402 | FAIL | 36.704 | 42.8320 [42.7421, 42.9219] | 1.1670 | FAIL |
| rotate | 2.560 | 1.7760 [1.7164, 1.8356] | 0.6937 | FAIL | 6.080 | 5.2768 [5.2679, 5.2857] | 0.8679 | PASS |
| gate/up | 51.552 | 51.0592 [51.0139, 51.1045] | 0.9904 | PASS | 490.336 | 505.8720 [505.7977, 505.9463] | 1.0317 | PASS |
| epilogue | 4.608 | 3.4624 [3.4446, 3.4802] | 0.7514 | FAIL | 5.536 | 4.4128 [4.3833, 4.4423] | 0.7971 | FAIL |
| down | 39.552 | 39.2096 [39.0976, 39.3216] | 0.9913 | PASS | 245.472 | 255.5520 [255.4279, 255.6761] | 1.0411 | PASS |
| combine | 3.072 | 2.9504 [2.9395, 2.9613] | 0.9604 | PASS | 4.288 | 4.1888 [4.1670, 4.2106] | 0.9769 | PASS |

U96 is one prospectively registered sensitivity case, shown for comparison; it is not selected or accepted as served routing. Full data for U8/U32/U64/U96/U128 are in receipts/harness-timings.json. Both complete six-stage validation gates fail.

## Per-kernel counters

One profiled instance per stage per case (n=1; no CI). Fourteen kernel replay passes, clocks/cache control none. Nsight warns that uncontrolled caches and clocks can affect results. No whole-model memory-backup warning appears; no model is in the diagnostic process. These are standalone observations, not validated served counters. HBM rates are decimal GB/s; percentages use Nsight sustained-peak definitions. Occupancy is active-SM achieved occupancy, not a fraction of the entire device.

| Case / stage | Occupancy % | HBM GB/s | HBM peak % | ALU busy % | Tensor busy % | Registers/thread | Static shared bytes | Limiter / inference |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C1 group | 20.660 | 0.000 | 0.000 | 0.041 | 0.000 | 32 | 128 | one-block grid; limited parallelism |
| C1 rotate | 5.114 | 0.000 | 0.000 | 1.198 | 0.000 | 21 | 0 | small grid / short kernel; resource limits alone do not establish critical limiter |
| C1 gate/up | 20.845 | 1110.701 | 15.525 | 58.037 | 13.511 | 128 | 32832 | registers and shared memory: four resident blocks, 25% theoretical |
| C1 epilogue | 1.575 | 0.000 | 0.000 | 1.024 | 0.000 | 40 | 0 | small grid / short kernel; resource limits alone do not establish critical limiter |
| C1 down | 10.534 | 795.581 | 11.117 | 35.508 | 8.403 | 128 | 32832 | registers and shared memory: four resident blocks, 25% theoretical |
| C1 combine | 9.594 | 0.000 | 0.000 | 0.646 | 0.000 | 48 | 16384 | small grid / short kernel; resource limits alone do not establish critical limiter |
| C16 synthetic U96 group | 6.789 | 0.011 | 0.000 | 0.024 | 0.000 | 32 | 128 | one-block grid; limited parallelism |
| C16 synthetic U96 rotate | 23.112 | 0.255 | 0.004 | 9.168 | 0.000 | 21 | 0 | small grid / short kernel; resource limits alone do not establish critical limiter |
| C16 synthetic U96 gate/up | 24.233 | 1569.065 | 21.913 | 71.851 | 16.722 | 128 | 32832 | registers and shared memory: four resident blocks, 25% theoretical |
| C16 synthetic U96 epilogue | 17.669 | 0.118 | 0.002 | 12.840 | 0.000 | 40 | 0 | small grid / short kernel; resource limits alone do not establish critical limiter |
| C16 synthetic U96 down | 23.266 | 1560.588 | 21.796 | 69.331 | 16.404 | 128 | 32832 | registers and shared memory: four resident blocks, 25% theoretical |
| C16 synthetic U96 combine | 35.686 | 0.244 | 0.003 | 8.171 | 0.000 | 48 | 16384 | small grid / short kernel; resource limits alone do not establish critical limiter |

Grouped kernels are consistent with substantial ALU pressure and resource-limited occupancy, with HBM below saturation. This is an inference, not a certified bottleneck attribution. Lowering registers alone cannot raise residency unless the shared-memory limit also changes. ALU busy cycles are not an instruction-count/issue-class ceiling for a strip transformation, and replay counters do not establish an attainable improvement. All thirty-six stage samples, including routing sensitivity, are in receipts/counter-summary.json.

## Bounds and candidate dispositions

| Mechanism | C1 bound | C16 bound | Candidate result / interval |
|---|---:|---:|---|
| host-readback, retained G3.1 maximum | 1.030183x | 1.028311x | Below 1.05; unrun, interval N/A |
| one-guard | 1.000000x | 1.000000x | No eligible guard; unrun, interval N/A |
| exl3-strips applicable bound | UNKNOWN | UNKNOWN | Validation/provenance and instruction ceiling gates fail; unrun, interval N/A |

The finite **ideal-deletion outer ceilings** from G3.1 are C1 1.563686x (maximum report 1.564183x) and C16 2.444200x (maximum 2.444573x), n=5 reports each. They assume deleting all EXL3 time with other work fixed. They are not attainable roofline bounds, speed estimates, candidate admissions or confidence intervals. No synthetic counter-to-served Amdahl projection is promoted to a valid bound.

No engine patch, fresh full-model champion measurement or challenger run was authorized by the failed admission gate. The prior G3.1 champion remains C1 81.783164 [81.780889,81.785439] and C16 272.650814 [272.589889,272.711740] tok/s (t95, n=5 rounds each). These are prior observations, not G3.2 remeasurements. There is no new model token-ID comparison in G3.2.

## Operations, attempts and deviations

Total LEFT shared reservation cost: 0.059134358 GPU-hours across 3 claim windows, including failed preflight, initial invalid timing attempt, preparation, waits and successful diagnostics. This is reservation accounting; hardware busy-hours were not metered. Minimum observed free memory was 29573 MiB, above the 16 GiB reserve. All claims are released. The daily was never unloaded, so restore time is N/A. LEFT daily, both peer services and production router IDs/start times remained unchanged; production configuration hash is unchanged.

The initial read-only preflight used the wrong peer daily container name and stopped before GPU work; claim released. The original CUDA captured-event timing method failed (error 1); no valid samples, retained and prospectively amended. The counter CSV parser first expected long form; it was corrected for wide-form columns and explicit units before interpretation. Harness wall_seconds is an absolute monotonic tick; cost uses guard/claim elapsed timestamps instead. No invalid attempt is counted as a scientific falsifier.

Monitoring deviation: the maximum observed record spacing was 0.326765s, exceeding the registered <=0.250s polling target. No request arrival was observed; abort latency was not exercised and no release SLA is certified. Read-only load and cumulative work counters stayed unchanged throughout retained cases. This limits the operational claim; it does not prove zero interference.

No production LiteLLM reload/restart/configuration edit, Spark model load, daily lifecycle operation, engine push, external post, paper or PR occurred. Only this preregistration repository is pushed. Raw operational snapshots and large reports stay private, with anonymous artifact hashes in receipts/raw-hashes.json.

## Next step

Obtain a prospectively authorized capture of exact per-layer expert membership and representative checkpoint/activation slices from the frozen served workload, then validate all stages and cache/locality behavior. That requires a new routing-capture plan or existing payloads; this run did not unload the daily to collect them. Add an instruction-class work/issue ceiling for the concrete strip transformation before attempting a bound or engine patch. Repair and test the monitor cadence before any next shared run.

Branches: SelfOptimize feat/w27-g3-2 (local implementation head supplied at handoff), engine ronanlabs/sm103-enable at 7c1bc0be3df413f43c65fb278699b629a44e7a90 unchanged, docs feat/w7-sm103 (report head supplied at handoff). Only preregistration/outcomes were pushed.

Sources: receipts/results.jsonl, receipts/harness-timings.json, receipts/counter-summary.json, receipts/bounds.json, receipts/operations-summary.json and receipts/candidate-ledger.jsonl.

Source head snapshots (before report-only packaging metadata): SelfOptimize implementation `beaf45d8076f9f64f6645dfa0de04cf70f0a5b83`; documentation report `87b3dd8246fab37284abf2efccab864aa78a098c`; unchanged engine `7c1bc0be3df413f43c65fb278699b629a44e7a90`.
