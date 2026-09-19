"""
Stage 2.1: Walk all CSS files in data/stage1/css/, extract token streams,
aggregate frequency counts with node-type breakdown.

Saves:
  data/parsed/stage2/token_freq.json   {token: {"count": N, "types": {type: N, ...}}}
  data/parsed/stage2/token_streams.jsonl  one line per file, list of (text, type) tokens
                                   (used by BPE training in stage2_build_vocab.py)

Usage:
  python stage2_extract_tokens.py
  python stage2_extract_tokens.py --limit 1000   # smaller run
"""

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from css_walker import walk_css, BPE_TYPES


def _sanitize(s: str) -> str:
    """Strip lone UTF-16 surrogates that can't round-trip through UTF-8."""
    return s.encode("utf-8", "replace").decode("utf-8")

CSS_DIR = Path("data/stage1/css")
OUT_DIR = Path("data/parsed/stage2")


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N files (for testing)")
    parser.add_argument("--css-dir", type=Path, default=CSS_DIR)
    parser.add_argument("--file-list", type=Path, default=None,
                        help="Text file of .css paths, one per line. Build the "
                             "vocab from the TRAIN shard only so token "
                             "frequencies never see holdout files.")
    parser.add_argument("--out", type=Path, default=OUT_DIR,
                        help="Output directory (default: data/parsed/stage2)")
    args = parser.parse_args()

    OUT_DIR = args.out
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.file_list:
        files = [Path(line.strip()) for line in
                 args.file_list.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    else:
        files = sorted(args.css_dir.glob("*.css"))
    if not files:
        print(f"No files found. Run stage1_download.py first.")
        return
    if args.limit:
        files = files[: args.limit]

    print(f"Extracting tokens from {len(files):,} files...")

    # Counters: token text -> {count, types: {type: count}}
    freq: dict[str, dict] = {}
    streams_path = OUT_DIR / "token_streams.jsonl"

    total_tokens = 0
    failed_files = 0

    with streams_path.open("w", encoding="utf-8") as stream_f:
        for path in tqdm(files, unit="file", desc="Extracting"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                tokens = walk_css(text)
                # Walker emits 3-tuples (text, type, context); sanitize text only.
                tokens = [(_sanitize(t), tt, c) for t, tt, c in tokens]
            except Exception:
                failed_files += 1
                stream_f.write("[]\n")
                continue

            stream_f.write(json.dumps(tokens, ensure_ascii=False) + "\n")
            total_tokens += len(tokens)

            for tok_text, tok_type, _ctx in tokens:
                entry = freq.get(tok_text)
                if entry is None:
                    entry = {"count": 0, "types": defaultdict(int)}
                    freq[tok_text] = entry
                entry["count"] += 1
                entry["types"][tok_type] += 1

    # Convert defaultdicts to regular dicts for JSON
    serializable = {
        tok: {"count": v["count"], "types": dict(v["types"])}
        for tok, v in freq.items()
    }

    freq_path = OUT_DIR / "token_freq.json"
    print(f"\nWriting frequency table ({len(serializable):,} unique tokens)...")
    with freq_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

    print(f"\n--- Stage 2 extraction complete ---")
    print(f"Files processed   : {len(files):,}")
    print(f"Failed files      : {failed_files:,}")
    print(f"Total tokens      : {total_tokens:,}")
    print(f"Unique tokens     : {len(serializable):,}")
    print(f"Streams written   : {streams_path}")
    print(f"Frequencies written: {freq_path}")

    # Print top-20 preview
    print("\nTop 20 tokens by frequency:")
    top = sorted(serializable.items(), key=lambda x: -x[1]["count"])[:20]
    for tok, v in top:
        primary_type = max(v["types"].items(), key=lambda x: x[1])[0]
        display = tok if len(tok) <= 40 else tok[:37] + "..."
        print(f"  {v['count']:>10,}  [{primary_type:>8s}]  {display!r}")


if __name__ == "__main__":
    main()
