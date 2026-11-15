# Amendment 5: preserve the native producer-to-EXL3 input locality

Before stage timing or candidate samples. Exact payload replay has passed a
byte-only early C1 smoke. The standalone cache scrub would evict the picks and
normalized activation, although in the native sequence their GPU producers
immediately precede grouping/rotation. Correct this known cache-state mismatch
before validation rather than interpreting scrub-cold metadata as served work.

After each 256 MiB scrub on the measured stream, copy only the captured pick
array and normalized input activation from persistent pinned host buffers on
that same stream. Then launch the six-stage graph. Put timing events after
these copies, never inside graph capture. The same preparation applies to
both A/B arms and all validation replays. Weights and scales are not prefetched;
all internal stage dependencies remain in the six-stage graph. Retain the
original exact bytes, geometry and arithmetic. Copies are preparation, not a
measured candidate operation. The transfer gate still requires all six stages
within the original 15% tolerance; no duration observations motivated a cache
parameter choice or threshold change. This locality approximation still needs
validation and does not establish a whole-model cache envelope.
