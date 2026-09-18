"""
Stage 4.0: Pre-encode the entire CSS corpus to compact mmap-able binary files.

Replaces the 8.86 GB token_streams.jsonl + per-load in-memory format with a
disk-resident, mmap-friendly layout that lets the dataset stream arbitrarily
large corpora without RAM pressure.

Outputs (written to data/parsed/stage4/):
  tokens.bin           uint32  array of all token IDs concatenated in file order
  file_offsets.bin     uint64  length N+1; file_offsets[i:i+1] bounds file i in tokens.bin
  anchors.bin          packed records {file_idx u4, pos u4, r f4, g f4, b f4} = 20 bytes
  anchor_offsets.bin   uint64  length N+1; anchor_offsets[i:i+1] bounds file i's anchors
  meta.json            human-readable summary

Usage:
  python stage4_preencode.py
  python stage4_preencode.py --limit 1000     # only first 1000 files (smoke)
  python stage4_preencode.py --out data/parsed/stage4_smoke   # custom dir
"""

from __future__ import annotations

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from color_detection import detect_colors
from css_walker import walk_css
from tokenizer import CSSTokenizer

CSS_DIR = Path("data/stage1/css")
DEFAULT_OUT = Path("data/parsed/stage4")

ANCHOR_DTYPE = np.dtype([
    ("file_idx", np.uint32),
    ("pos", np.uint32),
    ("r", np.float32),
    ("g", np.float32),
    ("b", np.float32),
    ("channel", np.int8),   # -1 = full (R/G/B), 0/1/2 = single channel
    ("_pad", np.uint8, 3),  # explicit padding to 24-byte alignment
])


def _sanitize(s: str) -> str:
    """Strip lone UTF-16 surrogates that can't round-trip through UTF-8."""
    return s.encode("utf-8", "replace").decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N files (for smoke runs)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output directory")
    parser.add_argument("--add-special", action="store_true", default=True,
                        help="Prepend BOS / append EOS per file (default true)")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = CSSTokenizer.load("data/parsed/stage2")
    print(f"Tokenizer: {tok}")

    files = sorted(CSS_DIR.glob("*.css"))
    if args.limit:
        files = files[: args.limit]
    print(f"Pre-encoding {len(files):,} files -> {out_dir}/")

    tokens_path = out_dir / "tokens.bin"
    file_offsets_path = out_dir / "file_offsets.bin"
    anchors_path = out_dir / "anchors.bin"
    anchor_offsets_path = out_dir / "anchor_offsets.bin"
    meta_path = out_dir / "meta.json"

    # Streaming writes so we never hold the full corpus in RAM
    file_offsets: list[int] = [0]
    anchor_offsets: list[int] = [0]

    total_tokens = 0
    total_anchors = 0
    total_value_context = 0      # for diagnostics: how many tokens are CTX_VALUE
    files_with_no_anchors = 0
    failed_files = 0

    with open(tokens_path, "wb") as tok_f, open(anchors_path, "wb") as anc_f:
        for file_idx, path in enumerate(tqdm(files, unit="file", desc="Encoding")):
            try:
                css = path.read_text(encoding="utf-8", errors="replace")
                stream = walk_css(css)
                # Sanitize text on tokens (handles lone surrogates from web CSS)
                stream = [(_sanitize(t), ty, c) for t, ty, c in stream]
                ids, stream_to_id = tok.encode_tokens_with_positions(
                    stream, add_special=args.add_special,
                )
                anchors, _hsls = detect_colors(stream)
            except Exception:
                failed_files += 1
                # Still record an empty file so file_idx aligns with our list
                file_offsets.append(file_offsets[-1])
                anchor_offsets.append(anchor_offsets[-1])
                continue

            ids_arr = np.asarray(ids, dtype=np.uint32)
            tok_f.write(ids_arr.tobytes())
            file_offsets.append(file_offsets[-1] + ids_arr.size)
            total_tokens += ids_arr.size

            # Diagnostic count
            total_value_context += sum(1 for _, _, c in stream if c == "value")

            # Build anchor records for this file
            if anchors:
                recs = np.empty(len(anchors), dtype=ANCHOR_DTYPE)
                for k, a in enumerate(anchors):
                    id_pos = stream_to_id[a.position]
                    ch = -1 if a.channel is None else int(a.channel)
                    recs[k] = (file_idx, id_pos, a.r, a.g, a.b, ch, (0, 0, 0))
                anc_f.write(recs.tobytes())
                anchor_offsets.append(anchor_offsets[-1] + recs.size)
                total_anchors += recs.size
            else:
                anchor_offsets.append(anchor_offsets[-1])
                files_with_no_anchors += 1

    # Write the offset tables
    np.asarray(file_offsets, dtype=np.uint64).tofile(file_offsets_path)
    np.asarray(anchor_offsets, dtype=np.uint64).tofile(anchor_offsets_path)

    # Meta
    meta = {
        "n_files": len(files),
        "failed_files": failed_files,
        "total_tokens": total_tokens,
        "total_anchors": total_anchors,
        "total_value_context_positions": total_value_context,
        "files_with_no_anchors": files_with_no_anchors,
        "add_special": bool(args.add_special),
        "anchor_dtype": str(ANCHOR_DTYPE.descr),
        "vocab_size": tok.total_size,
        "fat_head_size": tok.fat_size,
        "pad_id": tok.pad_id,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "files_size_mb": {
            "tokens.bin": tokens_path.stat().st_size / 1e6,
            "file_offsets.bin": file_offsets_path.stat().st_size / 1e6,
            "anchors.bin": anchors_path.stat().st_size / 1e6,
            "anchor_offsets.bin": anchor_offsets_path.stat().st_size / 1e6,
        },
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    # Report
    print(f"\n--- Pre-encode complete ---")
    print(f"Files processed     : {len(files):,}")
    print(f"Failed files        : {failed_files:,}")
    print(f"Total tokens (IDs)  : {total_tokens:,}")
    print(f"Total anchors       : {total_anchors:,}")
    print(f"  anchors / 1k IDs  : {1000 * total_anchors / max(1, total_tokens):.2f}")
    print(f"Files w/o anchors   : {files_with_no_anchors:,}")
    print(f"\nDisk usage:")
    for name, size_mb in meta["files_size_mb"].items():
        print(f"  {name:>20s}  {size_mb:>10.2f} MB")
    print(f"\nMeta: {meta_path}")


if __name__ == "__main__":
    main()
