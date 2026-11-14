"""Deterministic public workloads; tokenizer is a frozen input, never an inference call."""

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

SENT = "Report {i}: the morning was clear over sector {s}, winds were light, and the ferry ran on time as usual with no delays reported by the harbour office staff on duty.\n"


def digest(b):
    return hashlib.sha256(b).hexdigest()


def generate(tokenizer, output):
    tok = Tokenizer.from_file(str(tokenizer))
    lines = [SENT.format(i=i, s=i % 12) for i in range(2500)]
    source = tok.encode("".join(lines), add_special_tokens=False).ids
    rows = []
    for size in [4000, 8000, 16000, 32000, 48000, 60000]:
        for depth in [0.25, 0.5, 0.75]:
            for seed in [42, 43]:
                code = "N" + digest(f"{size}-{depth}-{seed}".encode())[:10].upper()
                at = int(size * depth)
                text = (
                    tok.decode(source[:at])
                    + f"\nNotice: the secret code for today is {code}. Remember it exactly.\n"
                    + tok.decode(source[at:size])
                )
                prompt = (
                    text
                    + "\nQuestion: What is the secret code in the text above? Answer with just the code."
                )
                rows.append(
                    {
                        "id": f"{size}-{depth}-{seed}",
                        "target_tokens": size,
                        "depth": depth,
                        "seed": seed,
                        "expected": code,
                        "prompt": prompt,
                        "expected_prompt_tokens": len(
                            tok.encode(prompt, add_special_tokens=False).ids
                        ),
                        "prompt_sha256": digest(prompt.encode()),
                    }
                )
    output.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    generate(a.tokenizer, a.output)
