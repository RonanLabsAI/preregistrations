# Amendment 3: preserve the served per-slot epilogue mode

Before any valid complete served payload or candidate timing. The sixth capture
reached the last stage and failed an invalid CUDA copy after five stage outputs
were copied. Source inspection confirms this family calls Experts.runSlots:
DownArgs.wts and DownArgs.out are null. The kernel writes per-slot FP32 y and
returns; the family performs its routing-weight reduction outside this sequence.
The earlier synthetic harness supplied nonnull weights and a combined output.

Preserve the actual nullable pointers. Copy optional weights/output only when
nonnull; retain and byte-compare the real per-slot y when out is null. Report
these six EXL3 stages with that exact served mode. The family reduction is
unchanged other work. No invented weights or combined output may replace it.
The failed capture is retained as a method attempt; restore and GPU accounting
are retained. All timing tolerances, candidate arithmetic and promotion rules
remain frozen. This corrects payload handling, not the frozen served workload.
