# Amendment 1: complete control cells in smoke

Frozen 2026-11-14 PT, before any G3.0 timed inference request. No timed
requests have yet run; the champion server is loading for smoke.

The smoke now begins with the complete uninstrumented champion remeasurement
(one excluded warm round plus five complete rounds at C1 and C16). This
supplies both matching drift gates before profiling and measures the longest
checkpoint block on the actual harness. It replaces the five first-prompt
plain blocks for the overhead comparator: the first prompt's five retained
native and client timings are extracted from the full C1 rounds. The traced
smoke remains five 512-token first-prompt blocks with exact reference IDs.

Measure overhead against those five first-prompt controls; compute the ratio
of means of native seconds (primary) and client seconds (secondary). An
increase above 10% in either stops the generation. Report descriptive t95
intervals for each arm and the paired log-ratio interval across ordered five
repeats; ordering is an operational screen and supports no promotion claim.
If drift fails, do not profile. If profiling fails or exceeds the cap, park,
restore, retain all outcomes and keep Mission Control execution held.

A passed smoke still requires actual park and timed daily restoration before
admission. A second acquisition remeasures matching complete control cells
again before generation measurements. No numerical band, candidate gate,
workload, budget or claim policy changes.
