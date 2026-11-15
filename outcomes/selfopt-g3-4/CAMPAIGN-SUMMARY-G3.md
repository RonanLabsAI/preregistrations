# SelfOptimize Generation 3 campaign summary, G3.0-G3.4

The campaign stops at G3.4 for PI review with no new champion; retain frozen
TensorFold 7c1bc0be on GB300. Null kernel results below do not constitute an
audited whole-model no-win result, and unrun gate dispositions are not falsifiers.

| Generation | What was learned | Decision |
|---|---|---|
| G3.0 / W24 | Whole-model counter replay adds about 485% native overhead and invokes memory backup; unprofiled controls stay in champion bands with identical IDs. | Stop before patching; those counters cannot justify a speed change. |
| G3.1 / W26 | Accepted timelines show EXL3 at 36.05% C1 / 59.09% C16, dense BF16 at 31.73% / 7.64%, device idle about 3%; counter probe still fails overhead cap. | Idle deletion upper below 1.05; host-readback and absent guard changes not admitted. |
| G3.2 / W27 | Isolated grouped kernels show 11-24% achieved occupancy, 11-22% HBM peak, 128 registers and 32,832 shared bytes; synthetic routing/complete transfer fails. | Counters suggest ALU/resource pressure but do not certify a decoder ceiling; obtain real payloads. |
| G3.3 / W28 | Real 288-expert/per-slot capture, middle C16 has 97 experts; native replay passes all stage checks. Three occupancy arms pass bitwise; best full EXL3 C1 t4r96 1.046897 [1.040771,1.053060], projecting only 1.016413 [1.014324,1.018500]. | Occupancy-only caps/tiles do not clear admission; no candidate model run. |
| G3.4 / W29 | Real captured slot batching already decodes once per expert/member tile; maximum four members fits sixteen. MCG scalar lookup is bitwise but much slower; paired rotation/slot epilogue has negligible full-sequence benefit. | All six projection interval uppers below 1.05; no candidate model run. Stop campaign. |

G3.4 best C16 stages projection is 1.001320 [1.000939,1.001702], n=10
paired blocks; C1 stages is 0.998064 [0.991337,1.004757]. Lookup projections
are about 0.395 C1 and 0.201 C16. Every measured arm/cell passes five bitwise
repeats. Native primary transfer passes all twelve checks; graph C1 rotation
continues to fail, so C1 graph projections remain diagnostic. Narrow intervals
do not bound transfer error or unseen layers/routing. Full numerical/resource
receipts are in the G3.4 result package.

Ruled out within these measured scopes: enough gain from deleting idle gaps,
these occupancy caps/tiles, additional within-step slot batching on the middle
capture, this scalar lookup decoder, and these rotation/slot epilogue changes.
Not ruled out: a different in-register decoder implementation, an alternative
format or on-chip predecoded tile strategy, a different routing/layer envelope,
or C1 dense BF16 work. Alternative formats/caches require memory, lifetime,
nonfinite, shape and full-model token gates; the current evidence does not make
them cheap or authorize them. No numerical tolerance was widened.

Recommendation (three sentences):

Stop this campaign and retain champion `7c1bc0be`; further occupancy caps, slot rebatching on these captures, and this scalar lookup do not justify another generation. If the PI continues, make it a separate instruction-guided C16 EXL3 study aimed at reducing in-register state extraction and codebook arithmetic while preserving MMA and reduction order. Budget an estimated 2-5 engineer-days and 1-2 GPU-hours for diagnosis, with up to 3.5 additional GPU-hours for full-model qualification only after a preregistered empirical gate passes.
