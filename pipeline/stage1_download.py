"""
Stage 1: Stream CSS files from bigcode/the-stack (v1), apply quality filters,
save 100k clean files to data/stage1/css/ with JSON metadata sidecars.

Prerequisites:
  - pip install -r requirements.txt
  - HuggingFace account with bigcode/the-stack terms accepted
  - HF_TOKEN environment variable set

Usage:
  python stage1_download.py
  python stage1_download.py --target 5000   # smaller run for testing
"""

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import json
import os
import sys
from pathlib import Path

import tinycss2
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

TARGET_DEFAULT = 100_000
SIZE_MIN = 100
SIZE_MAX = 1_024 * 1_024  # 1MB
MIN_DECLARATIONS = 5
OBFUSCATION_DECL_THRESHOLD = 500
OBFUSCATION_SINGLE_CHAR_RATIO = 0.5

OUT_CSS = Path("data/stage1/css")
OUT_META = Path("data/stage1/meta")

REQUIRED_FIELDS = {"content", "lang", "max_stars_repo_name", "max_stars_repo_path"}


# ---------------------------------------------------------------------------
# CSS quality checks
# ---------------------------------------------------------------------------

def _count_in_rules(rules) -> int:
    """Recursively count declarations across top-level rules, @media bodies,
    @supports bodies, @font-face bodies, and @keyframes bodies."""
    count = 0
    for rule in rules:
        if rule.type == "qualified-rule":
            decls = tinycss2.parse_declaration_list(rule.content, skip_whitespace=True)
            count += sum(1 for d in decls if d.type == "declaration")
        elif rule.type == "at-rule" and rule.content is not None:
            # Body could be either nested rules (@media, @supports, @keyframes)
            # or a flat declaration list (@font-face, @page).
            try:
                nested = tinycss2.parse_rule_list(
                    rule.content, skip_whitespace=True, skip_comments=True,
                )
                if nested and any(r.type in ("qualified-rule", "at-rule") for r in nested):
                    count += _count_in_rules(nested)
                else:
                    decls = tinycss2.parse_declaration_list(
                        rule.content, skip_whitespace=True,
                    )
                    count += sum(1 for d in decls if d.type == "declaration")
            except Exception:
                pass
    return count


def count_declarations(css_text: str) -> int:
    rules = tinycss2.parse_stylesheet(css_text, skip_whitespace=True, skip_comments=True)
    return _count_in_rules(rules)


def _collect_errors_in_rules(rules, out: list) -> None:
    for rule in rules:
        if rule.type == "error":
            out.append(getattr(rule, "message", "unknown"))
        elif rule.type == "qualified-rule":
            decls = tinycss2.parse_declaration_list(rule.content, skip_whitespace=True)
            out.extend(d.message for d in decls if d.type == "error")
        elif rule.type == "at-rule" and rule.content is not None:
            try:
                nested = tinycss2.parse_rule_list(
                    rule.content, skip_whitespace=True, skip_comments=True,
                )
                if nested and any(r.type in ("qualified-rule", "at-rule") for r in nested):
                    _collect_errors_in_rules(nested, out)
                else:
                    decls = tinycss2.parse_declaration_list(
                        rule.content, skip_whitespace=True,
                    )
                    out.extend(d.message for d in decls if d.type == "error")
            except Exception:
                pass


def collect_parse_errors(css_text: str) -> list[str]:
    rules = tinycss2.parse_stylesheet(css_text, skip_whitespace=True, skip_comments=True)
    errors: list[str] = []
    _collect_errors_in_rules(rules, errors)
    return errors


def has_parse_errors(css_text: str) -> bool:
    return len(collect_parse_errors(css_text)) > 0


def _collect_qualified_rules(rules, out: list) -> None:
    """Walk all qualified rules at any nesting level (top + inside @media etc.)."""
    for rule in rules:
        if rule.type == "qualified-rule":
            out.append(rule)
        elif rule.type == "at-rule" and rule.content is not None:
            try:
                nested = tinycss2.parse_rule_list(
                    rule.content, skip_whitespace=True, skip_comments=True,
                )
                _collect_qualified_rules(nested, out)
            except Exception:
                pass


def is_obfuscated(css_text: str) -> bool:
    rules = tinycss2.parse_stylesheet(css_text, skip_whitespace=True, skip_comments=True)
    qualified: list = []
    _collect_qualified_rules(rules, qualified)
    if not qualified:
        return False
    single_char = sum(
        1 for r in qualified
        if len("".join(
            t.value for t in r.prelude if hasattr(t, "value")
        ).strip()) == 1
    )
    total_decls = count_declarations(css_text)
    return (
        (single_char / len(qualified)) > OBFUSCATION_SINGLE_CHAR_RATIO
        and total_decls > OBFUSCATION_DECL_THRESHOLD
    )


def passes_quality(content: str) -> tuple[bool, str]:
    """Return (passes, rejection_reason)."""
    size = len(content.encode("utf-8", errors="replace"))
    if size < SIZE_MIN:
        return False, "too_small"
    if size > SIZE_MAX:
        return False, "too_large"
    if has_parse_errors(content):
        return False, "parse_error"
    if count_declarations(content) < MIN_DECLARATIONS:
        return False, "too_few_declarations"
    if is_obfuscated(content):
        return False, "obfuscated"
    return True, ""


# ---------------------------------------------------------------------------
# Schema probe
# ---------------------------------------------------------------------------

def probe_schema(dataset) -> dict:
    """Read the first record, verify required fields, return it."""
    record = next(iter(dataset))
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        print(f"\nERROR: Dataset is missing expected fields: {missing}")
        print(f"Actual fields: {list(record.keys())}")
        print("The Stack v2 schema may have changed. Update REQUIRED_FIELDS and field references.")
        sys.exit(1)
    print(f"Schema OK. Fields: {list(record.keys())}")

    # Detect stars field name
    for candidate in ("max_stars_count", "star_events_count", "stars"):
        if candidate in record:
            print(f"Stars field: '{candidate}'")
            return record, candidate
    print("WARNING: No stars field found. Star filtering will be skipped.")
    return record, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT,
                        help="Number of CSS files to collect (default: 100000)")
    parser.add_argument("--min-stars", type=int, default=10,
                        help="Minimum repo stars (default: 10)")
    args = parser.parse_args()

    # Auth check
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("Steps to fix:")
        print("  1. Create an account at https://huggingface.co")
        print("  2. Accept terms at https://huggingface.co/datasets/bigcode/the-stack")
        print("  3. Generate a token at https://huggingface.co/settings/tokens")
        print('  4. Set it: $env:HF_TOKEN = "hf_..."  (PowerShell)')
        sys.exit(1)

    # Create output dirs
    OUT_CSS.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    # Check resume state
    existing = sorted(OUT_CSS.glob("*.css"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Resuming: {start_idx} files already saved, collecting {args.target - start_idx} more.")
    if start_idx >= args.target:
        print(f"Already have {start_idx} files. Done.")
        return

    # Load CSS-specific parquet files directly (avoids scanning all languages)
    print("Listing CSS parquet files...")
    from datasets import load_dataset
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    parquet_files = sorted(fs.ls("datasets/bigcode/the-stack/data/css", detail=False))
    hf_paths = [f"hf://datasets/bigcode/the-stack/data/css/{p.split('/')[-1]}"
                for p in parquet_files]
    print(f"Found {len(hf_paths)} CSS parquet files.")

    dataset = load_dataset(
        "parquet",
        data_files={"train": hf_paths},
        streaming=True,
        split="train",
        token=token,
    )

    # Probe schema
    first_record, stars_field = probe_schema(dataset)

    # Reload (probe consumed the first record via iter)
    dataset = load_dataset(
        "parquet",
        data_files={"train": hf_paths},
        streaming=True,
        split="train",
        token=token,
    )

    # Filter by minimum stars (lang is always CSS here)
    def keep(x):
        if stars_field and (x.get(stars_field) or 0) < args.min_stars:
            return False
        return True

    filtered = dataset.filter(keep)

    # Counters
    rejected = {"too_small": 0, "too_large": 0, "parse_error": 0,
                "too_few_declarations": 0, "obfuscated": 0, "exception": 0}
    saved = start_idx
    seen = 0

    pbar = tqdm(total=args.target, initial=start_idx, unit="file", desc="Collecting CSS")

    size_samples = []

    for record in filtered:
        if saved >= args.target:
            break

        seen += 1
        content = record.get("content", "")

        try:
            ok, reason = passes_quality(content)
        except Exception as e:
            rejected["exception"] += 1
            continue

        if not ok:
            rejected[reason] += 1
            continue

        idx = saved
        css_path = OUT_CSS / f"{idx:07d}.css"
        meta_path = OUT_META / f"{idx:07d}.json"

        try:
            css_path.write_text(content, encoding="utf-8")
            meta = {
                "repo_name": record.get("max_stars_repo_name", ""),
                "path": record.get("max_stars_repo_path", ""),
                "stars": record.get(stars_field, 0) if stars_field else None,
                "size": len(content.encode("utf-8", errors="replace")),
                "lang": record.get("lang", "CSS"),
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            # Partial write — clean up
            css_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            rejected["exception"] += 1
            continue

        saved += 1
        size_samples.append(meta["size"])
        pbar.update(1)

        # Disk estimate after first 100 files
        if saved == 100 and size_samples:
            avg_bytes = sum(size_samples) / len(size_samples)
            est_gb = (avg_bytes * args.target) / 1e9
            pbar.write(f"Disk estimate: ~{est_gb:.1f} GB for {args.target:,} files "
                       f"(avg {avg_bytes/1024:.1f} KB per file)")

    pbar.close()

    print(f"\n--- Stage 1 complete ---")
    print(f"Files saved : {saved:,}")
    print(f"Records seen: {seen:,}  (includes pre-filter CSS+stars check)")
    print(f"\nRejection breakdown:")
    for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
        if count:
            print(f"  {reason}: {count:,}")

    if saved < args.target:
        print(f"\nWARNING: Only collected {saved:,} of {args.target:,} target files.")
        print("The dataset may be exhausted or filters are too strict.")

    print(f"\nNext step: python stage1_verify.py")


if __name__ == "__main__":
    main()
