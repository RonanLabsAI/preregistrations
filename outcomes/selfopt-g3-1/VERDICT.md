# Generation result

Accepted low-overhead timelines; no candidate met every applicability and bound gate. No code candidate ran, no speedup is claimed, and no audited scientific null is counted.

All numerical results below are sourced from `receipts/results.jsonl`, with the conservative sensitivity separately in `receipts/host-bound-sensitivity.json`.

Champion C1: mean 81.783164 tok/s, t95 CI [81.780889, 81.785439], n=5 retained complete rounds after an excluded warm round; original drift and CV gates pass.
Champion C16: mean 272.650814 tok/s, t95 CI [272.589889, 272.711740], n=5 retained complete rounds after an excluded warm round; original drift and CV gates pass.

C1 is native per-reply decode; C16 is aggregate client throughput. Their metrics are separate. All measured output IDs match frozen references, n=364 replies including excluded warm-ups.

C1 timeline overhead: native 0.451121%, client 0.452755%, n=5 blocks/arm; both pass the frozen cap. Native ordered ratio CI [1.003200, 1.005823], an operational comparison rather than a randomized causal estimate.
C16 timeline overhead: native 0.164147%, client 0.162520%, n=5 blocks/arm; both pass the frozen cap. Native ordered ratio CI [1.000939, 1.002344], an operational comparison rather than a randomized causal estimate.

One sampled counter reply: native 43.895295s versus control mean 6.248157s, ratio 7.025319x; n=1 profiled reply versus n=5 native controls, no ratio CI available. It fails the separate cap and its counters cannot inform candidate bounds.

C1 median device-idle fraction 2.917318%, EXL3 union 36.048525%, dense BF16 union 31.734682%; n=5 reports. Family unions can overlap and are not an additive time budget.
C16 median device-idle fraction 2.679098%, EXL3 union 59.086817%, dense BF16 union 7.638512%; n=5 reports. Family unions can overlap and are not an additive time budget.

C1 traces use the preregistered first prompt; C16 uses full-width shared rounds, n=5 reports per cell, each resolving 511 cycles after prefill.

Host-readback maximum observed free-idle ceiling: C1 1.030183x and C16 1.028311x, n=5 reports per cell; below the admission bar and unrun.

Host-readback sensitivity ceilings remain below the admission bar, C1 1.037719x and C16 1.034469x, n=5 reports per cell; this post-analysis robustness check deliberately adds all DtoH API time and the positive overhead interval upper estimate to idle.

EXL3 strips remains unrun with unknown applicable decoder ALU bound; rejected replay counters, occupancy, MMA and STREAM are insufficient.

Guard widening remains unrun and not applicable; no eligible predicate in the current source audit (n=1 static source audit).

LEFT exclusive reservation 0.554575 GPU-hours, n=1 reservation window including load, profiling, CPU waits and restore; shared resident fallback GPU busy time was not metered.

Preserved daily restoration 101.697s to health, 103.121s to both routed LEFT answers, n=1 manual restore cycle. Claims released, research route removed, production and RIGHT identities unchanged.

This was a PI-authorized manual implementation run with local durable checkpoints; no new Mission Control product was queued/admitted and no MC heartbeat or automatic interruption guarantee is claimed.

Next: preregister a served-shape EXL3 decoder diagnostic that avoids whole-model replay backup, then derive a finite applicable bound before any patch.
