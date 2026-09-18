"""
Stage 2.3: Round-trip test.

For a sample of CSS files:
  1. Parse CSS -> token stream A
  2. Encode A -> IDs
  3. Decode IDs -> token stream B
  4. Verify concat(A) == concat(B) on text values

This catches bugs in fat-head lookup, BPE routing, BPE decoding, and
any token-text mutation along the way. Type tags may differ on the
decoded side (BPE returns "bpe" for sub-words), so we only compare text.

Usage:
  python stage2_test_roundtrip.py
  python stage2_test_roundtrip.py --sample 5000
"""

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from css_walker import walk_css
from tokenizer import CSSTokenizer

CSS_DIR = Path("data/stage1/css")
TOK_DIR = Path("data/parsed/stage2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1000,
                        help="Number of files to round-trip (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading tokenizer from {TOK_DIR}...")
    tok = CSSTokenizer.load(TOK_DIR)
    print(f"  {tok}")

    files = sorted(CSS_DIR.glob("*.css"))
    if len(files) > args.sample:
        random.seed(args.seed)
        files = random.sample(files, args.sample)
    print(f"Round-tripping {len(files):,} files...")

    n_ok = 0
    n_fail = 0
    failure_examples: list[tuple[str, str, str]] = []
    total_ids = 0
    total_tokens_in = 0

    for path in tqdm(files, unit="file", desc="Round-trip"):
        try:
            text = path.read_text(encoding="utf-8")
            stream_in = walk_css(text)
            ids = tok.encode_tokens(stream_in)
            stream_out = tok.decode(ids)
        except Exception as e:
            n_fail += 1
            if len(failure_examples) < 5:
                failure_examples.append((str(path), "EXC", str(e)))
            continue

        # Walker emits 3-tuples (text, type, context); decoder emits 2-tuples.
        text_in = "".join(tok[0] for tok in stream_in)
        text_out = "".join(tok[0] for tok in stream_out)

        total_tokens_in += len(stream_in)
        total_ids += len(ids)

        if text_in == text_out:
            n_ok += 1
        else:
            n_fail += 1
            if len(failure_examples) < 5:
                # Find first diverging position
                diff_pos = next(
                    (i for i, (a, b) in enumerate(zip(text_in, text_out)) if a != b),
                    min(len(text_in), len(text_out)),
                )
                start = max(0, diff_pos - 30)
                end_in = min(len(text_in), diff_pos + 30)
                end_out = min(len(text_out), diff_pos + 30)
                failure_examples.append((
                    str(path),
                    f"DIFF @ pos {diff_pos}",
                    f"  in:  ...{text_in[start:end_in]!r}\n  out: ...{text_out[start:end_out]!r}",
                ))

    print(f"\n--- Round-trip report ---")
    print(f"Files tested      : {n_ok + n_fail:,}")
    print(f"Passed            : {n_ok:,}  ({n_ok / max(1, n_ok+n_fail):.1%})")
    print(f"Failed            : {n_fail:,}  ({n_fail / max(1, n_ok+n_fail):.1%})")
    print(f"Total tokens in   : {total_tokens_in:,}")
    print(f"Total ids out     : {total_ids:,}")
    if total_tokens_in:
        print(f"IDs per token     : {total_ids / total_tokens_in:.2f}")

    if failure_examples:
        print(f"\nFirst {len(failure_examples)} failures:")
        for path, kind, detail in failure_examples:
            print(f"  {path}  [{kind}]")
            print(f"  {detail}")

    if n_fail == 0:
        print("\nOK: tokenizer round-trips cleanly. Proceed to Stage 3.")
    else:
        print(f"\nWARNING: {n_fail} round-trip failures. Investigate before Stage 3.")


if __name__ == "__main__":
    main()
