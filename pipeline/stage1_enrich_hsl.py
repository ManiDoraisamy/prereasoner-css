"""
Enrich the corpus with real hsl()-bearing CSS from the-stack.

The base corpus has hsl() in only ~0.9% of files (914 of 100k). This script
streams the SAME source (bigcode/the-stack, CSS split) with the SAME quality
filters as stage1_download.py, but keeps ONLY files that use hsl()/hsla(),
skipping anything already in the base corpus (by content hash). No synthetic
CSS, no rewriting — just more of the naturally-occurring hsl-using files
that the original uniform download undersampled.

The base download's stream order is deterministic per dataset snapshot, so
the base corpus is a prefix of what this scan sees; the content-hash dedup
makes re-scanning it harmless, just slow. Files land in a SEPARATE directory
(data/stage1_enrich/css) so the base corpus stays byte-identical to the
original run; make_splits.py merges the two pools explicitly via --extra-dir
and records the provenance in split_meta.json.

Best-effort by design: stops at --target files OR --max-minutes, whichever
comes first. The experiment is viable without it (make_splits.py stratifies
whatever exists); this just raises the hsl file pool.

Usage:
    python pipeline/stage1_enrich_hsl.py --target 3000 --max-minutes 120
"""

from __future__ import annotations

# --- repo path shim ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from stage1_download import passes_quality, probe_schema

BASE_CSS = Path("data/stage1/css")
OUT_CSS = Path("data/stage1_enrich/css")
OUT_META = Path("data/stage1_enrich/meta")


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def build_dedup_set() -> set[str]:
    """Content hashes of the base corpus + any previously enriched files."""
    seen: set[str] = set()
    for d in (BASE_CSS, OUT_CSS):
        if not d.exists():
            continue
        files = sorted(d.glob("*.css"))
        for f in tqdm(files, desc=f"Hashing {d}", unit="file"):
            try:
                seen.add(_hash(f.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=3000,
                        help="hsl-bearing files to collect (default 3000)")
    parser.add_argument("--min-stars", type=int, default=10)
    parser.add_argument("--max-minutes", type=float, default=180,
                        help="Wall-clock budget; stop cleanly when exceeded")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set (put it in .env). See stage1_download.py.")
        sys.exit(1)

    OUT_CSS.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    existing = sorted(OUT_CSS.glob("*.css"))
    saved = len(existing)
    if saved >= args.target:
        print(f"Already have {saved} enriched files. Done.")
        return

    print("Building dedup set (base corpus + prior enrichment)...")
    seen_hashes = build_dedup_set()
    print(f"  {len(seen_hashes):,} known content hashes")

    print("Listing CSS parquet files...")
    from datasets import load_dataset
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    parquet_files = sorted(fs.ls("datasets/bigcode/the-stack/data/css",
                                 detail=False))
    hf_paths = [f"hf://datasets/bigcode/the-stack/data/css/{p.split('/')[-1]}"
                for p in parquet_files]
    print(f"Found {len(hf_paths)} CSS parquet files.")

    def make_stream():
        return load_dataset("parquet", data_files={"train": hf_paths},
                            streaming=True, split="train", token=token)

    _first, stars_field = probe_schema(make_stream())
    dataset = make_stream()

    def keep(x):
        if stars_field and (x.get(stars_field) or 0) < args.min_stars:
            return False
        return True

    filtered = dataset.filter(keep)

    rejected = {"no_hsl": 0, "duplicate": 0, "quality": 0, "exception": 0}
    seen = 0
    t0 = time.time()
    deadline = t0 + args.max_minutes * 60

    pbar = tqdm(total=args.target, initial=saved, unit="file",
                desc="Collecting hsl CSS")

    for record in filtered:
        if saved >= args.target:
            break
        if time.time() > deadline:
            pbar.write(f"Time budget ({args.max_minutes} min) reached; "
                       f"stopping with {saved} files.")
            break

        seen += 1
        content = record.get("content", "") or ""
        low = content.lower()
        if "hsl(" not in low and "hsla(" not in low:
            rejected["no_hsl"] += 1
            continue

        try:
            ok, _reason = passes_quality(content)
        except Exception:
            rejected["exception"] += 1
            continue
        if not ok:
            rejected["quality"] += 1
            continue

        h = _hash(content)
        if h in seen_hashes:
            rejected["duplicate"] += 1
            continue
        seen_hashes.add(h)

        css_path = OUT_CSS / f"e{saved:06d}.css"
        meta_path = OUT_META / f"e{saved:06d}.json"
        try:
            css_path.write_text(content, encoding="utf-8")
            meta_path.write_text(json.dumps({
                "repo_name": record.get("max_stars_repo_name", ""),
                "path": record.get("max_stars_repo_path", ""),
                "stars": record.get(stars_field, 0) if stars_field else None,
                "sha1": h,
                "source": "the-stack css split, hsl-filtered enrichment",
            }, ensure_ascii=False), encoding="utf-8")
        except Exception:
            css_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            rejected["exception"] += 1
            continue

        saved += 1
        pbar.update(1)

    pbar.close()
    elapsed = (time.time() - t0) / 60
    print(f"\nScanned {seen:,} records in {elapsed:.1f} min; saved {saved:,} "
          f"hsl files to {OUT_CSS}/")
    print(f"Rejected: {rejected}")
    print("\nNext: python pipeline/make_splits.py --extra-dir data/stage1_enrich/css")


if __name__ == "__main__":
    main()
