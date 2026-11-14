"""Render MATH-500 eval prompts with the fork's own `qwen` template (run in venv-train).

The user turn is the problem followed by the training data's exact boxed-answer suffix
(`BOXED_USER_SUFFIX` in example/scripts/build_dataset.py), so evaluation prompts are
formatted exactly as training instructions were. Output: token ids per problem.
"""

import hashlib
import json
import sys
from pathlib import Path

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from transformers import AutoTokenizer

sys.path.insert(0, "/work/CIG-SFT-public/example/scripts")
from build_dataset import BOXED_USER_SUFFIX


def main() -> int:
    model_dir, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    tok = AutoTokenizer.from_pretrained(model_dir)
    template = get_template_and_fix_tokenizer(tok, DataArguments(template="qwen"))
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    with out.open("x") as f:
        for i, r in enumerate(rows):
            messages = [{"role": "user", "content": r["problem"] + BOXED_USER_SUFFIX}, {"role": "assistant", "content": ""}]
            prompt_ids, _ = template.encode_oneturn(tok, messages)
            f.write(json.dumps({"index": i, "unique_id": r["unique_id"], "answer": r["answer"],
                                "prompt_ids": prompt_ids}) + "\n")
    print(json.dumps({"n": len(rows), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "example": tok.decode(json.loads(out.read_text().splitlines()[0])["prompt_ids"])[:400]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
