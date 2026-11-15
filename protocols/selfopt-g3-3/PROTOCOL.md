# SelfOptimize G3.3: empirical grouped EXL3 occupancy experiments

Prospective W28 manual experiment. All timestamps use America/Los_Angeles (PT).
Engine parent 7c1bc0be; SelfOptimize parent fac50d4a. Only this public repository
may be pushed. Captured weights, activations, raw operational snapshots and
profiler databases stay private; publish anonymous hashes and aggregate results.

## Capture and transfer gate

The daily uses a different runtime. Permit one W26-style exclusive champion
capture window, with claimed resident fallback checks, preserved daily stop/start
and timed restore. Use the identical champion binary, checkpoint revision
078455ffe6472f9a52fbc1139f58b9db2881b25c, frozen G3.1 recipe and prompt/token arrays.
A diagnostic driver-call interposer copies one complete six-stage sequence for
C1 and exact full-width C16, including picks, grouping outputs, all scalar launch
arguments, active checkpoint trellises, scales, activations and stage outputs.
Capture is intrusive and supplies payloads only, never performance samples.
Select the first complete sequence after an explicit arm marker for each width;
no selection by observed performance. Verify frozen long output token IDs.
Collect native timelines separately with capture disarmed: Nsight Systems CUDA
node tracing only, CPU sampling/context switches off, no Nsight Compute replay.
Use G3.1's accepted five-report served stage distributions as the initial target;
retain fresh served traces and report changes. C16 is actual routing, never a
synthetic unique-expert substitute. Record layer/sequence selection limitations.

Replay all six stages from the captured inputs with original binary/geometry.
Use one excluded warm block and five retained blocks of twenty replays, scrub
256 MiB outside timing. Kernel timestamps come from Nsight Systems. Report stage
block medians, mean and t95 (df4, critical 2.776445105); every mean must be within
15% of the served target. Report every failure without widening tolerance.
If graph scheduling fails transfer, a native-launch sensitivity may be recorded
separately; it cannot silently replace the frozen primary design. Captured
stage outputs must reproduce bitwise. Any incomplete provenance/transfer gate
prevents applicable end-to-end inference; standalone candidate results remain
explicitly diagnostic. No ceiling is needed to implement these bounded A/Bs.

## Candidates, frozen before timing

Only MCG grouped EXL3 on sm_103, preserving all existing sm_120+ code/selection.
Separate opt-in symbols leave the champion selected until numerical admission.
All keep four warps, identical K partition, MMA and warp-reduction order, FP32
accumulators and the existing split counts. Candidate set:

1. r96: original eight N tiles, maximum 96 registers/thread. Shared-memory
   residency remains a possible limiter; spills may falsify the mechanism.
2. t4r96: four N tiles (64 columns), maximum 96 registers/thread, twice the
   N-axis CTAs and half the reduction shared storage.
3. t2r64: two N tiles (32 columns), maximum 64 registers/thread, four times the
   N-axis CTAs and quarter reduction shared storage.

Evaluate both gate/up and down, C1 and C16. Compile failures and resource growth
are outcomes. Require byte equality of every grouped partial and complete
six-stage output against champion on the captured inputs, finite outputs and
five repeat runs. A byte mismatch rejects that candidate; no new tolerance.
Existing full kernel contracts, edges, nonfinite, FP64 and planted controls are
required before any end-to-end candidate serving. Diagnostic-only symbols are
not declared production-qualified by a microbenchmark.

## Alternating warm kernel A/B and empirical admission

For each case/candidate run one excluded warm pair then ten retained paired
blocks, twenty repetitions per arm. Alternate AB/BA by block. Same inputs,
stream, launch geometry except the registered N tile, and 256 MiB cache scrub
before each replay. Use native CUDA event time around the complete six-stage
sequence (outside graph capture); also retain Nsight per-stage timestamps.
Report paired log speed ratio with two-sided t95 (df9, critical 2.262157163),
per-stage and full-six-stage ratios, and all block values. No post-hoc pruning.
A/B result is the finite empirical bound, conditional on unchanged other work:
B=1/(1-f+f/r), f C1=0.360485, C16=0.590868 from G3.1. Use full-six-stage r,
so unchanged EXL3 stages and their overhead are included. Transform both interval
ends and report point/lower/upper. No occupancy or ideal-deletion ceiling is an
attainable speed estimate. Numerical pass and bound point >=1.05 in either cell
admit end-to-end work only after the complete transfer/contract gates pass.

## Safety, monitor and budgets

Every GPU action/probe requires a live scoped preemptible fleet claim. No Spark
loads, production router restart/reload/config edit, clock changes, or key output.
Shared harness only: free memory >=24 GiB, allocations <=4 GiB, >=16 GiB free
reserve plus 4 GiB replay reserve. Initial daily idle gate: zero running/waiting/
pending for ten seconds, unchanged cumulative prefill/decode counters.
Dedicated monitor uses persistent local HTTP connections and NVML memory queries,
monotonic deadlines at 100ms, timeout <=100ms, a separate <=250ms watchdog, and
claim heartbeat <=3s. Log actual start/completion cadence. Any arrival, read
failure, reserve/claim/heartbeat/cadence breach kills only the owned process tree.
Test injected arrival, load failure, claim expiry and stalled monitor before GPU
work; report measured detection/termination latency, never a scheduler SLA.
Stop after an actual production arrival. Single case <=120s; shared diagnostic
reservation <=30 GPU-minutes. Capture/restore <=40 GPU-minutes; full-model phase
<=210 GPU-minutes including restoration, stop new work ten minutes before cap.

## End-to-end and promotion

Use the W26 preserved-daily lifecycle and resident text/image fallback checks.
Champion first, five alternating fresh-load champion/candidate pairs, one warm
and five retained rounds per C1/C16 cell. Frozen exact prompt/token arrays,
512-token replies, draft=false, thinking off, temperature 0, seed 1234, cache off,
context 40960, parallel 16. C1 mean native 511/seconds; C16 8192/client seconds.
Champion original bands [80.12335390502187,83.39369488073706] and
[267.3277037430555,278.23903858971084], CV<=1.5%; drift halts scoring.
Promotion requires identical token IDs, point >=1.05x, paired lower >1,
candidate throughput interval wholly above contemporaneous champion, other
cell paired lower >=0.98, and cache-off 8K/32K prefill max of five <=1.10x.
A passing candidate is the new champion. Report all failures/nulls/attempts,
GPU reservation hours and timed daily health plus both routed answers. Engine
commits stay local on ronanlabs/sm103-enable; SelfOptimize feat/w28-g3-3 off
feat/w27-g3-2; documentation feat/w7-sm103. No upstream publication implied.
