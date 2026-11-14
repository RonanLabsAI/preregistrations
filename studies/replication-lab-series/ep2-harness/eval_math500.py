"""MATH-500 Avg@16 / Pass@16 with vLLM and math-verify (run in venv-eval inside the container).

16 samples per problem, temperature 1.0, top-p 1.0, max 4,096 new tokens, per-problem
sampling seed derived from the unit seed. Resumable: problems already in the output
file are skipped. Output: one JSON line per problem with per-sample correctness.
"""

import argparse
import json
import sys
from pathlib import Path

from math_verify import parse, verify
from vllm import LLM, SamplingParams

N_SAMPLES, TEMPERATURE, TOP_P, MAX_NEW = 16, 1.0, 1.0, 4096
CONTEXT = 4096  # Qwen2.5-Math-1.5B max_position_embeddings; generation is capped at the remaining context
STOP_TOKEN_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>: the qwen template turn end, identical for all arms


def grade(text: str, answer: str) -> bool:
    try:
        gold = parse("$" + answer + "$")
        pred = parse(text)
        return bool(pred) and bool(verify(gold, pred))
    except Exception:  # noqa: BLE001 - any parser/verifier failure grades the sample incorrect
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["index"] for line in args.out.read_text().splitlines() if line.strip()}
    todo = [r for r in rows if r["index"] not in done]
    print(json.dumps({"event": "eval-start", "model": args.model, "todo": len(todo), "done": len(done)}), flush=True)
    if not todo:
        return 0
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=CONTEXT, gpu_memory_utilization=0.85, seed=args.seed,
              enable_prefix_caching=True)
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        params = [SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=min(MAX_NEW, CONTEXT - len(r["prompt_ids"])), stop_token_ids=STOP_TOKEN_IDS,
                                 seed=args.seed * 100003 + r["index"]) for r in chunk]
        outs = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in chunk], params, use_tqdm=False)
        with args.out.open("a") as f:
            for r, o in zip(chunk, outs):
                samples = [{"correct": grade(c.text, r["answer"]), "tokens": len(c.token_ids),
                            "finish": c.finish_reason} for c in o.outputs]
                f.write(json.dumps({"index": r["index"], "unique_id": r["unique_id"], "samples": samples}) + "\n")
        print(json.dumps({"event": "eval-chunk", "done": len(done) + start + len(chunk)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
