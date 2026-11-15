# Prospective diagnostic binding

Frozen 2026-11-14 PT before hardware-counter or ceiling execution. These are
G3.0 profiling diagnostics, not code candidates or served speed claims.

After the valid timeline matrix, measure the same served champion with
Nsight Compute 2026.1.0: exact tf_exl3_grouped_1 filter, skip 256 matching
launches to avoid early decode, collect five matching launches. Sections:
LaunchStats, Occupancy, SpeedOfLight, ComputeWorkloadAnalysis,
MemoryWorkloadAnalysis and WarpStateStats. Kernel replay, cache control none,
clock control none. Administrative counter access is scoped to this profiler;
no driver/module or production setting changes. Use a separate scratch cache,
the exact binary and the frozen first prompt; verify its exact output IDs
before reading counters. Record server timing overhead separately. Replay
timings cannot become candidate speedups. Missing/unsupported counters remain
unknown. Excessive profiling overhead remains an operational stop.

The attached independent mma_roofline.cu diagnostic measures the applicable
mma.sync.m16n8k16 f16-input, f32-accumulator instruction ceiling on sm103a.
Four independent accumulators, eight 256-thread blocks per SM, 8192
iterations and ten launches per sample; five samples after an excluded warm
launch and exact constant-input checksum. The checksum depth would expose
reduced-precision accumulation saturation. Compile as a separate executable
with the installed toolkit, record that compiler identity, and never replace
any champion kernel or model data. Report throughput samples and descriptive
intervals. This is a physical MMA diagnostic, not proof of a dequant ALU
ceiling, HBM floor, whole-model bound, or a candidate promotion.
