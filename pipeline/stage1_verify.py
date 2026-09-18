"""
Stage 1 verification: run tinycss2 over all saved CSS files and report parse health.

Usage:
  python stage1_verify.py
  python stage1_verify.py --sample 1000   # verify a random sample instead of all files
"""

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from stage1_download import collect_parse_errors, count_declarations

CSS_DIR = Path("data/stage1/css")
FAILURE_RATE_WARN = 0.05  # warn if >5% fail


def classify_errors(css_text: str) -> list[str]:
    """Return all parse-error messages (top-level + nested in @media / @supports
    / @font-face / declaration values)."""
    return collect_parse_errors(css_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Verify a random sample of N files instead of all")
    args = parser.parse_args()

    all_files = sorted(CSS_DIR.glob("*.css"))
    if not all_files:
        print(f"No CSS files found in {CSS_DIR}")
        print("Run stage1_download.py first.")
        return

    files = all_files
    if args.sample and args.sample < len(all_files):
        files = random.sample(all_files, args.sample)
        print(f"Sampling {len(files):,} of {len(all_files):,} files")
    else:
        print(f"Verifying all {len(files):,} files")

    total = 0
    failed = 0
    error_counter: Counter = Counter()

    for path in tqdm(files, unit="file", desc="Verifying"):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            failed += 1
            error_counter[f"read_error: {type(e).__name__}"] += 1
            total += 1
            continue

        errors = classify_errors(text)
        total += 1
        if errors:
            failed += 1
            for msg in errors:
                # Truncate long error messages for grouping
                key = msg[:80] if msg else "unknown"
                error_counter[key] += 1

    failure_rate = failed / total if total else 0

    print(f"\n--- Stage 1 Verification Report ---")
    print(f"Total files checked : {total:,}")
    print(f"Files with errors   : {failed:,}  ({failure_rate:.1%})")
    print(f"Clean files         : {total - failed:,}  ({1 - failure_rate:.1%})")

    if error_counter:
        print(f"\nTop error messages (up to 10):")
        for msg, count in error_counter.most_common(10):
            print(f"  [{count:5,}]  {msg}")

    print()
    if failure_rate > FAILURE_RATE_WARN:
        print(f"WARNING: Parse failure rate {failure_rate:.1%} exceeds {FAILURE_RATE_WARN:.0%} threshold.")
        print("Investigate before proceeding to Stage 2.")
        print("Common causes: encoding issues, CSS variables with unusual syntax,")
        print("  @charset declarations, or tinycss2 parser limitations.")
    else:
        print(f"OK: Parse failure rate {failure_rate:.1%} is within the {FAILURE_RATE_WARN:.0%} threshold.")
        print("Proceed to Stage 2.")

    print(f"\nHuman checkpoint: manually inspect 10-20 random files.")
    print(f"  python -c \"import random, pathlib; "
          f"[print(p) for p in random.sample(sorted(pathlib.Path('data/stage1/css').glob('*.css')), 20)]\"")


if __name__ == "__main__":
    main()
