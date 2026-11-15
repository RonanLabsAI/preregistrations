# Execution binding

Recorded 2026-11-14 PT before post-admission generation measurement.
SelfOptimize implementation: local source commit
82896f37e3deace53eb6a4b9479577305365ca8a. Entry point:
`python3 rung2/gen3/execute.py`. The real smoke exercised its collector,
profile and lifecycle functions. Subsequent non-timing changes add resumed
block selection and Mission Control heartbeat/checkpoint reporting; requests,
frozen arrays, token gates, clocks and statistical rules are unchanged.

The admitted interruption contract uses at most ten minutes to release and
at most two minutes of lost work, supported by the completed smoke and its
longest observed block of 102.996794 seconds. The smoke record reports the
actual restore observation; it does not claim an automatic scheduler release
guarantee. The combined reservation budget still includes smoke, subsequent
measurement, waiting, shutdown and restore.
