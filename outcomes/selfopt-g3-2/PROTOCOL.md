# SelfOptimize G3.2: isolated EXL3 decoder diagnostic

Prospective manual W27 experiment, authorized under the approved generation
and Station-use template. Frozen before any diagnostic CUDA execution.
All times use America/Los_Angeles (PT). Publish all outcomes and attempts;
only the preregistration repository may be pushed. No posting or paper is implied.

## Identity and provenance gate

Use the original champion EXL3 experts fatbin, SHA-256 in binding.json, without
compilation or kernel modification. Engine 7c1bc0be, SelfOptimize parent fae16338.
Public G3.1 outcome 50ce1848 supplies valid timeline evidence and frozen model
controls. The six decoder stages are grouping, rotate-in, grouped gate/up,
gate/up epilogue, grouped down, and down/combine. Exact grids, register and
shared-memory metadata and five served-duration distributions are retained in
served-durations.json. Dimensions 4096, expert width 2048, eight slots, 256 experts,
MCG four-bit trellises, gate/up split K=4, down split K=1, block sizes unchanged.

Important pre-execution limitation: retained G3.1 CUDA timelines record launch
geometry and time, not input expert-ID/member arrays. No captured C16 routing
payload has been located. C1's eight distinct top-k picks fix its member-count
pattern; expert labels can be renamed but synthetic weights/data are not the
original checkpoint. C16 uses a labeled balanced routing sensitivity sweep with
8,32,64,96,128 distinct experts, each row having eight distinct picks. This is
NOT the captured routing distribution and NOT an exhaustive routing envelope.
The exact served-distribution validation gate cannot pass without payload
provenance. Do not silently replace missing data or claim an applicable bound.
If payloads become available, publish a prospective amendment before using them.
No whole model may be loaded just to obtain routing under this protocol.

## Standalone harness and validation

harness.py uses the CUDA driver to load the exact binary and packed kernel ABI,
and allocates only expert trellises and per-layer scratch, no model or KV cache.
Weights are deterministic LCG seed 1234; inputs/scales are finite fixed values.
Every output must be finite and byte-repeatable; grouping must return the
specified unique count. These checks validate the diagnostic, not model accuracy.
Original engine qualification remains the numerical reference.

Use a captured six-stage CUDA graph with timing events to avoid Python launch
latency in stage times. Scrub 256 MiB before each timed replay, outside timing.
One excluded warm block, five retained blocks of twenty replays each. Report
block medians, mean and two-sided t95 (df4, critical 2.776445105), and resources.
Compare each stage's mean of block medians with the median of the five
served-report medians: every stage within +/-15% is necessary. Report failures,
not just overall agreement. Values, weight locality, graph scheduling, clock
and interference differences can invalidate transfer even at identical shapes.
C16 timing agreement alone never substitutes for routing provenance.
No widening tolerances after observing data.

## Counters and finite bounds

After timing, take one isolated six-kernel profile per routing case, using
LaunchStats, Occupancy, SpeedOfLight, MemoryWorkloadAnalysis and ComputeWorkloadAnalysis
with clock/cache control none, kernel replay, profile-from-start off. A permission
failure permits scoped sudo-n for the profiler, not driver-setting changes.
Report achieved occupancy, theoretical/resource occupancy, register/thread,
shared-memory limits, HBM GB/s and percent of peak, ALU and tensor utilization.
Retain profiler warnings and replay passes. Replay timing is never native timing.
No whole-model memory backup occurs: process allocations are capped at 4 GiB.
Counters remain standalone-only when validation fails.

For a validated unchanged-work mechanism, define local optimistic roofline
ceiling r <= 1/max(u_HBM,u_ALU,u_tensor), where utilizations are fractions of
Nsight's corresponding peak, and the candidate preserves their work/traffic.
Use only a peak that actually bounds its operation class; general SM utilization,
occupancy, STREAM and dense MMA are not a decoder ALU ceiling. If applicable
instruction-class evidence is unavailable, the decoder bound remains unknown.
A change that removes instruction work needs a separate instruction-count bound.
Occupancy ratios alone cannot admit a candidate. Derive model ceilings as
B=1/((1-f)+f/r), using G3.1 union fractions, never summed overlapping time.
Also report the finite ideal-deletion outer ceiling 1/(1-f), explicitly distinct
from an attainable roofline/candidate bound. EXL3 union fractions are C1 0.360485
and C16 0.590868 (medians); include maximum-report sensitivity.

Candidate exl3-strips is only a smaller grouped decoder tile/register footprint
on sm103, preserving sm120+ behavior. AHEAD is excluded. Do not implement before
validated finite applicable bound >=1.05. host-readback retains G3.1's <=1.030184
C1 / <=1.028312 C16 bound; one-guard retains 1.0; neither is eligible here.
Absent admission, close as a method gate result, not an audited scientific null.

## Shared daily safety and budgets

All GPU work, including tiny probes, requires a live UUID-scoped preemptible
fleet claim. Use LEFT beside its preserved resident daily only if free memory
is >=24 GiB before context/allocation, with <=4 GiB diagnostic allocation,
>=16 GiB free reserve and another 4 GiB replay reserve. Recheck before every
case. No full-model diagnostic, unrelated model or Spark load. No clocks changed.

Read-only daily load endpoints must show zero running, waiting and pending
requests for 10 seconds before starting. Monitor at <=250ms intervals while
running, including cumulative prefill/decode counters to detect completed
requests. Any arrival, load read failure, memory reserve breach, claim loss or
watchdog failure stops owned diagnostic processes. Record observed detection
and release latency; do not claim zero interference or automatic scheduler SLA.
Claim liveness is checked on the controller and supplied via a <=3s heartbeat;
no heartbeat means abort. Counter runs may use root only inside the guarded
owned process tree. Stop no production process. A single case has a 120s
wall cap, diagnostic phase <=30 GPU-minutes including waits; stop after a
production arrival rather than immediately resuming. Retain interrupted attempts.

No daily unload for diagnostic validation or routing capture. Only an admitted
candidate permits a LEFT exclusive full-model window. Before that, perform
W26's claimed resident fallback checks; stop/start the preserved daily using
its own Docker lifecycle, restore it and time health plus both routed answers.
Production LiteLLM configuration, identity and start time must remain unchanged.
Never restart/reload it, read its key to output, or edit its configuration.
Raw operations, large reports and binaries stay private on scratch; stat before rm.

## Conditional candidate measurement

Before candidate implementation, prospectively publish its exact surface,
work/traffic evidence, finite bound, falsifier and tests. Code on engine branch
ronanlabs/sm103-enable, local commits only; sm120+ behavior preserved. Require
existing seeded FP64, byte identity, served shapes/edges, nonfinite, repeatability,
negative-control and resource contracts before model serving. Champion FIRST.
Reuse G3.1 frozen model, recipe, public prompts, prefill arrays and exact IDs.
At least five alternating fresh-load champion/candidate pairs, one excluded
warm round and five retained rounds per C1/C16 cell. C1 = mean reply 511/native
seconds, C16=8192/client wall seconds. Original bands [80.12335390502187,
83.39369488073706] and [267.3277037430555,278.23903858971084]; CV<=1.5%.
Drift halts scoring; do not widen bands. Report t95 and paired log ratio t95.
Promotion: identical IDs, point >=1.05x, paired lower >1, throughput interval
wholly above contemporaneous champion, other cell paired lower>=0.98, cache-off
8K/32K prefill max of five <=1.10x champion. Such a passing candidate becomes
new champion; report plainly. Two executed falsifier failures close audited null.
Model phase <=210 GPU-minutes including restoration; stop new work 10 minutes
before cap. Production preempts; request timeout 900s. Heartbeat/checkpoint every
block. No new Mission Control product or automatic interruption guarantee claimed.
