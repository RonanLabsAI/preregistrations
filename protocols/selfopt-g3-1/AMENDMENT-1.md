# Amendment 1: uncaptured launch cycles

Frozen before any G3.1 timeline or counter request, during champion startup.
Source inspection of the frozen GLM lane adapter shows `queueFn` returns
`NotPipelined`; its verify path submits CUDA work directly and returns host
sampled tokens. A read-only inspection of the retained G3.0 schema confirms
CUDA driver APIs reside in the Nsight runtime table and may contain direct
kernel launches rather than `cuGraphLaunch`.

The prospective cycle attribution therefore accepts BOTH captured graph
launch cycles and uncaptured kernel-launch cycles. For the latter, anchors are
correlated DtoH output reads on the serving thread, their preceding stream
synchronization, and the next forward's HtoD/launch sequence. C1 must expose
511 post-prefill output-read boundaries per reply; if extra control reads occur,
disambiguate by byte size, stream, thread and source before accepting windows.
For C16, identify shared-round readbacks and exclude prompt/setup/tail periods;
report how many outputs belong to each round, or label per-reply attribution
unknown. The whole-round timeline still supports descriptive totals.

No cap, workload, numerical bar, promotion criterion, or budget changes. Do
not infer a graph-capture candidate from direct launch count alone; it requires
finite critical-path bounds and a separate prospective amendment.
