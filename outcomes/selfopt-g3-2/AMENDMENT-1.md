# Amendment 1: stage timing from Nsight Systems

Frozen after the first diagnostic attempt's timing API failure and before any
valid G3.2 timing sample. The original harness launched its six-stage sequence
and passed the initial finite-output and unique-count checks, then CUDA returned
error 1 when elapsed time was requested from captured graph timing events. No
stage timing or counters were obtained. The daily stayed loaded and unchanged;
the shared reservation was released. Retain that attempt as invalid.

Use the same original fatbin, six-stage graph, cold-cache preparation and
one warm plus five retained blocks of twenty replays. Remove captured timing
events. Collect Nsight Systems CUDA kernel start/end timestamps, with graph
nodes, CPU sampling and context-switch tracing OFF. This matches the served
reference's per-kernel timing observable. Export the report to SQLite and
extract the 120 graph replays in order; the initial six-stage non-graph warm
sequence is excluded. Each block supplies twenty samples per stage. Block 0
is excluded, blocks 1-5 retained. Report their stage medians and t95 as frozen.
No profiler replay duration substitutes for native stage time.

No tolerance, shape, routing limitation, candidate gate, memory reserve,
production-priority rule or GPU budget changes. Updated harness hash is in
binding.json; the original is preserved in the initial public commit.
