# Prospective STREAM diagnostic

Frozen 2026-11-14 PT before any ceiling diagnostic executes. The independent
mma_roofline.cu also measures a CUDA STREAM triad: three disjoint 2 GiB f32
arrays, C=A+2B, ten full-array launches per sample, five samples after excluded
initialization/warm-up. Count two reads and one write, 12 bytes per element.
Inputs are exact constants; verify first/last output values equal five before
timing. Arrays exceed cache capacity; this is a streaming bandwidth reference,
not a claim about random EXL3 accesses or model throughput. Report every sample,
mean/t95 interval and the minimum observed rate as a conservative within-window
reference floor. It is not a guaranteed population floor. No clock changes,
no model changes, one claimed GB300. Free all allocations on completion.
The original MMA gate and five samples remain unchanged.
