# Amendment 4: fixed middle-decode primary capture

Before harness validation or candidate timing. The first valid early C1/C16
payloads reproduce the frozen workload's exact token IDs. Early C16 has only
12 distinct experts, a genuine initial sequence rather than the steady routing
used by the G3.1 stage-duration aggregate. Retain it as an early-route control;
do not select or reject it by performance, which has not been measured.

Add one prospective primary payload at a fixed middle position: after arming
for each width, skip 10,731 matching full-width, eight-slot group launches and
capture the next complete six-stage sequence. With 42 routed layers this is
the 22nd layer of the 256th matching full-width step (one-based indexing).
The count is fixed now, independent of its unique count, values or speed.
Count matching launches, not elapsed time or observed performance. The ordinal
is relative to full-width groups after arming; no assumption that a partial
prefill/tail is a decode step is needed for that definition.

Use this fixed middle primary for the frozen validation and A/B design, retain
the early payload and its hashes, and report both routing counts. The primary
is one layer/step, not a distribution-wide or worst-case envelope. Preserve the
same G3.1 initial duration targets and 15% tolerance; any transfer failure remains
visible and prevents applicable end-to-end admission. No workload, token IDs,
candidate, interval, safety or promotion threshold changes.
