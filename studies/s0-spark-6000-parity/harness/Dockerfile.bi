# S0 serving image: the pinned vLLM nightly with batch-invariant mode on. ENV only, no file layer.
# Built reproducibly on each host (see protocol): SOURCE_DATE_EPOCH=0, rewrite-timestamp, no attestations.
FROM vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063
ENV VLLM_BATCH_INVARIANT=1
