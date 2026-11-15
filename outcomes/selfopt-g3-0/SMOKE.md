# G3.0 profiled smoke and admission

Recorded 2026-11-14 PT after prospective protocol commits 8f1c7a5 and
959225a. Source: SMOKE.json. This is an operational smoke, not a
candidate improvement or an audited scientific null.

Both matching five-round champion cells pass their original bands:
C1 native reply decode 81.791330 [81.790090, 81.792570] tok/s;
C16 aggregate client 272.689967 [272.638686, 272.741247] tok/s.
Intervals are t95 over five complete round values, df4. All exact token
references match, including one excluded warm round per cell.

Five traced first-prompt blocks pass exact output equality. Native timing
overhead is 0.6301%, ordered ratio interval [1.002507, 1.010101]; client
overhead is 0.6231%, ordered ratio interval [1.002553, 1.009915]. These
ordered smoke comparisons cannot promote a candidate. The trace contains
3665070 CUDA kernel events. Summed kernel family duration is not a
critical-path share or a bandwidth/occupancy counter. GPU timeline gaps
remain an upper bound, not demonstrated removable host bubble.

Checkpointing and actual park/restore were exercised. Health restoration
took 145.788679 seconds; both production routes answered from the restored
daily by 147.210155 seconds. The stop helper initially misclassified an
exited process with an empty command line, then found the profiling session
already shut down; the GPU was free. Both failed cleanup invocations are
retained as operational observations. The repaired free-device confirmation
passed. No device result was discarded or pooled away.

The product was admitted through the approved standing template. No Mission
Control code change or deployment occurred. Production routing and the
other serving Station retained their original identities. Automatic
Station parking remains disabled. Full generation measurement and every
conditional candidate disposition will be recorded separately.
