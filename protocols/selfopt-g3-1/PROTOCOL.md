# SelfOptimize G3.1: low-overhead decode paths and conditional candidates

Prospective successor to the stopped G3.0. Frozen before G3.1 GPU requests.
The PI approved the generation template and taking Station time for TensorFold.
This is a manual execution under revocable device claims; no scheduler guarantee
is inferred. Every attempt, null, gate stop, and unrun disposition is retained.

## Identity and controls

The champion, model revision, server hash, recipe, exact public corpus, prefill
arrays, and output-ID references are unchanged from ../selfopt-g3-0. Binding.json
pins their hashes and the original control bands. Verify every hash before runs.
One rank, GB300 sm103, stored EXL3, BF16 activations/KV, FP32 accumulation,
context 40960, parallel 16, prompt cache zero, thinking/draft OFF, temperature
zero, seed 1234. Sixteen public prompts, exactly 512 outputs per decode reply.
8192/32768 input arrays generate sixteen outputs and characterize prefill only.

Remeasure the uninstrumented champion FIRST: one excluded warm round and five
retained full rounds each at C1/C16. C1 = round mean of 511/native decode seconds;
C16 = 8192/client round wall seconds. Means must be in the unchanged C1
[80.12335390502187,83.39369488073706] and C16
[267.3277037430555,278.23903858971084] bands; sample CV <=1.5%.
Failure halts scoring. Never widen a band. All token IDs must match references.
Report descriptive t95 intervals over five rounds (df4, critical 2.776445105).

## Separate instrumentation gates

Timeline: Nsight Systems CUDA tracing with graph nodes; CPU sampling, context
switch tracing, and backtraces OFF. No per-kernel hardware counters. Launch the
exact champion under an interactive profiler; exclude a warm reply, then five
first-prompt C1 replies. Compare native decode and client wall ratio of means
against the SAME first prompt's five fresh native controls above. Either increase
>5% rejects the timeline and dependent bounds. Also report ordered paired log
ratio t95 (an operational comparison, not a randomized causal estimate).
If valid, collect five C16 whole-round traces, with inactive tracing rounds
interleaved, and compare BOTH to native C16 controls and inactive rounds under
the same 5% cap. First C1 trace runs untraced then traced; subsequent C1 samples
alternate inactive/traced to observe injection overhead separately. Native-server
controls remain mandatory. All warm-up/setup/export time is excluded from reply
time but included in reservation cost. Retain first trace and all failures.

Counters: AFTER acceptable timeline collection, at most ONE selected
`tf_exl3_grouped_1` instance after skipping 256 matching launches on ONE frozen
first-prompt reply. Nsight Compute LaunchStats and Occupancy only; clock/cache
control none; kernel replay; separate scratch cache. No full-model section set.
Compare its native and client times to the five native same-prompt controls;
either increase >10% rejects counter evidence. Collect no second counter reply
if rejected. Never use replay duration as native performance or use rejected
counters for a candidate bound. Counter rejection does NOT discard separately
valid timelines. No extra MMA/STREAM run: those are not dequant ALU ceilings.

## Attribution and bounds, frozen before patches

For each timeline, use CUDA stream IDs and API correlation IDs. Identify decode
iterations by the repeated graph-launch -> dependent output copy -> host wait ->
next graph cycle, excluding prefill, first token, setup and between-request idle.
Report cycle counts, per-cycle wall time, union of device-active intervals,
stream overlap, kernel-family union/sum, copy bytes/times, graph API cost and
synchronization latency. Kernel sums may overlap and are NEVER Amdahl shares.
If complete per-reply correlation fails, label the window/metric unknown.

host-readback: use a conservative outer bound B=1/(1-b), b = within-decode
GPU-idle time fraction. This can REJECT a candidate below 1.05 but cannot admit
one by itself. Admission additionally requires correlated host work independent
of the awaited token (demonstrated by source dependency and CPU/API evidence),
and a measured removable critical-path fraction. No useful independent work =>
not applicable. If overhead >= half b, host evidence is inconclusive. Never
count dependent graph execution inside a host synchronization as host headroom.

exl3-strips: B=1/((1-f)+f/r), using EXL3's union on the correlated decode path,
not summed overlapping kernel durations. r must be a measured applicable
register-decoder instruction/ALU ceiling at served shapes, with occupancy and
resource checks; occupancy alone is insufficient. A missing applicable ceiling
means UNKNOWN, not infinity and not permission to patch. The ceiling diagnostic,
if needed, must be publicly amended with code, checksum, shapes, and budget
BEFORE execution. W14b's rejected EXL3 AHEAD is excluded.

one-guard: only an actually closed sm103 capability predicate on the active
served path; qmmf's already open predicates and EXL3 AHEAD are excluded.
B=1/(1-fslow), treating only its correlated slow-path interval as free. No
eligible guard => B=1.0, unrun N/A. Static code reachability is mandatory.

New timeline-discovered mechanisms: publish a dated prospective amendment
containing exact single surface, source dependency argument, measured path
share, finite attainable local speed estimate, B, correctness tests, falsifier,
metric, budget, and immutable implementation identity BEFORE any candidate
patch/GPU test. No selecting favorable output lengths or combining changes.
Every candidate whose valid bound >=1.05 and applicability gates pass runs.
An unknown bound or unsupported mechanism does not clear a gate.

## Candidate scoring and operational limits

Each executed candidate: >=5 alternating fresh-load champion/candidate pairs,
full frozen C1 and C16 workloads separately; exact IDs first. Kernel patches
require the existing seeded FP64, replacement byte identity, served-shape/edge,
nonfinite, repeatability, negative-control, and resource gates before serving.
Report raw five-round values, means/t95 and paired log speed ratios/t95.
Promotion requires point >=1.05, paired CI lower >1.00, candidate throughput
interval wholly above the matching contemporaneous champion interval,
CV <=1.5%, and five cache-off prefill samples at EACH length with empirical
p95=max at n5 <=1.10x contemporaneous champion. C16 must not regress if C1
promotes and vice versa (paired ratio lower >=0.98). Original ±2% bands remain
drift checks, not a significance test or a required 5% gain over their upper end.
Token change => token-changing, no promotion. Two consecutive executed
falsifier failures close an audited null. No admitted candidate => method result,
not a scientific null. A real passing speedup becomes a champion candidate.

Total LEFT reservation cap 240 GPU-minutes including load, CPU waits and restore;
candidate sub-cap 120. Stop new work by minute 230. Per-request timeout 900s;
checkpoint and claim check each reply/round. Production preempts. Stop only
owned research processes. Restore preserved daily with its own Docker lifecycle,
time health and both routed answers, compare production and RIGHT identities,
then release claims. Fallback checks run under shared claims for resident RIGHT
devices. No production router reload/restart/config edit; no unrouted Spark load.
Large outputs remain on scratch. Only this public preregistration repository may
be pushed, after private-content audit. Public packages exclude private host/path,
network, device UUID, credential, email and operational-identifying content.
