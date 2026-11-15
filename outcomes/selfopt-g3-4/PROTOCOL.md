# SelfOptimize G3.4: EXL3 work reduction, final generation before PI review

Prospective W29 manual run under the approved template. Times are PT
(America/Los_Angeles). SelfOptimize parent 42bdb25 (feat/w28-g3-3), engine
experimental parent 8209d5011; champion remains 7c1bc0be. Prior evidence:
[W28 report](../../outcomes/selfopt-g3-3/REPORT.md), G3.1 timeline and G3.2 counters.
Question: can reducing repeated MCG decode arithmetic or slot-stage setup earn
at least 1.05x whole-model decode without changing numerical results?

## Frozen arms and structural dispositions

A is the exact champion fatbin. Three B arms, with no tuning after timing:
1. lookup: replace MCG decode2 arithmetic by a 65,536-entry uint16 lookup,
   built once using the champion decode2 function, 128 KiB resident. Gate/up
   only; down remains champion. Same eight N tiles, four warps, K splits,
   MMA and warp-reduction order, maximum 128 registers. Table generation and
   transfer are one-time model setup, excluded from steady-state decode timing;
   resource cost and packed-half consistency must be reported. Any model-level
   attempt must include setup cost in load reporting and preserve the table.
2. stages: fuse the two input-rotation matrices in one CTA, loading the shared
   unrotated activation once while retaining both scales and fixed butterfly
   orders; use a captured-mode slot-only down epilogue with no reduction shared
   buffer. Null routing-weight and combined-output pointers are mandatory.
   Neither grouped projection changes.
3. combined: lookup and stages together, without further changes.

The source already decodes each expert once per sixteen-member tile. Captured
C16 has 128 slots, 97 distinct experts, maximum multiplicity four; C1 has eight
singletons. Extra expert batching cannot remove decode work here. Gate/up
have different trellis tables and different rotated inputs, so sharing a single
weight decode between them is invalid. These are structural unrun dispositions,
not measured speedup/nulls. No cross-step decoded-weight cache or format change.

## Inputs, numerics and transfer

Reuse W28's prospectively selected middle C1/C16 captures (group ordinal 10,732,
relative full-width step 256/layer 22; actual 288 expert table entries). Their
anonymous manifest SHA-256 is bound in binding.json. Checkpoint revision
078455ffe6472f9a52fbc1139f58b9db2881b25c, frozen G3.1 recipe, prompts, seed 1234,
cache off, temperature zero and long output token IDs remain fixed. Captured
weights, activations, routing, raw pointers and profiler databases stay private.
Use no random synthetic replacement or routing selection by observed speed.

Exact champion replay must reproduce every captured stage output bitwise.
Each B must match all stage outputs and grouped partials bitwise on five repeated
complete sequences, all final floats finite. No tolerance widening. LUT builder
checks every possible state in both packed-half positions. This does not replace
full nonfinite/shape/planted-control/precision contracts before model serving.
Any mismatch rejects its arm/cell and is recorded; no timing interpretation there.

Primary transfer validation is native launch, explicitly chosen prospectively
because G3.3's C1 graph rotation failed. One excluded warm block and five retained
blocks of twenty; Nsight Systems stage medians, t95 df4 critical 2.776445105.
Every mean must be within +/-15% of G3.1 served stage medians, all twelve cells.
Also retain graph transfer as a separate sensitivity. A graph failure limits
corresponding graph A/B applicability; it cannot be silently replaced. Standalone
A/Bs may still be diagnostic. No fresh capture or daily unload for this phase.

## Paired kernel design and gate

For each arm at C1 and C16: one excluded warm AB pair, then n=10 retained
alternating AB/BA block pairs, twenty replays per arm/block, no pruning.
Preparation scrubs 256 MiB and restores only picks/normalized input from pinned
buffers on the same stream outside timing. Weights/scales/table are not prefetched.
CUDA events surround the complete six-stage graph replay outside capture;
Nsight Systems CUDA node tracing records all stages (CPU sampling/context switches
off, no Nsight Compute). Report both per-stage and complete-sequence speed.
Use log(A block median / B block median), mean +/- t95 df9 critical 2.262157163,
then exponentiate. Paired intervals describe repeats, not transfer uncertainty.

For complete six-stage speed r, empirical conditional Amdahl projection:
B=1/(1-f+f/r), f_C1=0.360485 and f_C16=0.590868 (G3.1 measured EXL3 shares).
Transform point and both interval endpoints. Unchanged stages and overhead stay
in r; an ideal-deletion ceiling is not a candidate speed estimate. Admission
requires bitwise pass, applicable transfer and projected interval LOWER >1.05
in either C1 or C16. If transfer fails in an otherwise promising cell, halt at
that gate and record the exact limitation; do not call it an end-to-end null.

## Operations and budgets

Every GPU action, including lookup initialization and any probe, needs a scoped
preemptible fleet claim. Shared LEFT daily remains resident. Initial idle gate:
zero running/waiting/pending for ten seconds and unchanged cumulative prefill,
decode and busy counters. Initial free >=24 GiB, explicit harness allocation
<=4 GiB, free reserve >=16 GiB. Persistent local HTTP and NVML, monotonic 100 ms
deadlines, <=250 ms watchdog, <=3 s verified claim heartbeat. Any arrival, read
failure, allocation/reserve, cadence or claim failure terminates only owned
processes. Stop new GPU work after a real arrival. Inject arrival/read-failure/
claim-expiry/stall CPU tests before GPU work. Single GPU case <=120 seconds;
shared reservation <=30 GPU-minutes, stop new work after 28 minutes.
No production LiteLLM restart/reload/config edit or key output, no Spark loads,
clock changes, engine/SelfOptimize/docs pushes, external messages or posting.
Only this public preregistration/outcomes repository may be pushed after audit.
Large private artifacts stay in scratch. Stat before removal; preserve attempts.

## Conditional model phase

Only a passing empirical admission permits W26 preserved-daily lifecycle on LEFT,
with scoped fallback claims and verified resident text/image fallbacks. Remeasure
champion first. Five alternating fresh-load champion/candidate pairs, one warm and
five retained rounds per cell, 512 tokens, draft=false, thinking off, cache off,
context 40960, parallel16, seed1234, temperature0, frozen exact input/output IDs.
C1 metric 511/native seconds; C16 8192/client seconds. Champion bands fixed at
[80.12335390502187,83.39369488073706] and [267.3277037430555,278.23903858971084],
CV<=1.5%; drift halts. Require token-ID identity, point>=1.05, paired lower>1,
candidate throughput interval wholly above contemporaneous champion, other cell
paired lower>=0.98 and cache-off 8K/32K prefill max of five <=1.10x.
Full kernel contracts are mandatory before candidate serving. Full-model budget
<=210 GPU-minutes including restoration, stop new work ten minutes before cap.
A passing candidate is declared plainly as the new champion; otherwise retain
7c1bc0be. Preserve daily container/config/mounts, restore and time health plus
both routed answers; verify peers and production identity/config unchanged.

## Outcomes and stop

Publish every attempt, numerical/compile/transfer failure, per-arm/cell intervals,
projections, admissions, resource costs, GPU reservation-hours and restoration.
No candidate admission means no candidate model run and no fresh champion claim.
Write a G3.0-G3.4 campaign summary in outcomes and local campaign documentation:
what was learned, what is ruled out within measured scope, remaining direction
and likely cost. Then stop for PI review; do not begin G3.5. Local branches:
engine ronanlabs/sm103-enable, SelfOptimize feat/w29-g3-4 off feat/w28-g3-3,
docs feat/w7-sm103. Only preregistration and outcome records are published.
