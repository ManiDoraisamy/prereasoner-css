"""
Build the stratified train shard and the held-out probe set.

Why this exists — two measured problems with the original Phase 0 setup:

1. hsl() appears in only ~0.9% of the corpus (914 of 100,000 files). Training
   samples ~3% of corpus tokens, so the model that produced the emergence
   result saw on the order of a few hundred hsl() calls, total. The train
   shard here includes EVERY training-side hsl file, raising hsl exposure
   by an order of magnitude without a single synthetic example.
2. The original probes drew examples from the same files the model trained
   on. Here the holdout files are excluded from the train shard entirely,
   so every probe number is computed on CSS the model never saw.

Stratification changes the training DISTRIBUTION, not the training DATA —
every file is a real file from the corpus, and the composition is recorded
in split_meta.json for the paper.

Usage:
    python pipeline/make_splits.py                     # defaults
    python pipeline/make_splits.py --extra-dir data/stage1_enrich/css
Outputs:
    data/splits/train_files.txt
    data/splits/holdout_files.txt
    data/splits/split_meta.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def classify(path: Path) -> str:
    """'hsl' if the file uses hsl()/hsla(), else 'rgbfunc', else 'other'."""
    try:
        data = path.read_bytes().lower()
    except OSError:
        return "unreadable"
    if b"hsl(" in data or b"hsla(" in data:
        return "hsl"
    if b"rgb(" in data or b"rgba(" in data:
        return "rgbfunc"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--css-dir", type=Path, default=Path("data/stage1/css"))
    parser.add_argument("--extra-dir", type=Path, default=None,
                        help="Additional CSS dir (e.g. hsl-enriched download). "
                             "All its hsl files join the pools before splitting.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Shuffle seed for the split (NOT a training seed).")
    parser.add_argument("--holdout-hsl-frac", type=float, default=1 / 3,
                        help="Fraction of hsl files held out for probing.")
    parser.add_argument("--holdout-rgb", type=int, default=1500,
                        help="rgb()-bearing files held out.")
    parser.add_argument("--holdout-other", type=int, default=5000,
                        help="Plain files held out (hex/named/leakage probes).")
    parser.add_argument("--train-rgb-quota", type=int, default=6000,
                        help="rgb()/rgba()-bearing files in the train shard. "
                             "rgba() alone is in ~41%% of the corpus, so "
                             "'include all' would swamp the shard — a quota "
                             "keeps rgb() anchors plentiful without diluting "
                             "hsl density.")
    parser.add_argument("--train-size", type=int, default=14000,
                        help="Total train-shard file count. Deliberately "
                             "SMALLER than the corpus: training consumes a "
                             "fixed ~20M tokens (5000 steps x batch 4 x ctx "
                             "1024), so a denser shard means the model "
                             "actually SEES more hsl()/rgb() during those "
                             "steps. All train-side hsl files are mandatory; "
                             "rgb files fill to the quota; the rest is a "
                             "random fill of plain files.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()

    files = sorted(args.css_dir.glob("*.css"))
    if args.extra_dir:
        files += sorted(args.extra_dir.glob("*.css"))
    if not files:
        raise SystemExit(f"No CSS files under {args.css_dir}")

    pools: dict[str, list[Path]] = {"hsl": [], "rgbfunc": [], "other": []}
    for f in tqdm(files, desc="Classifying", unit="file"):
        kind = classify(f)
        if kind != "unreadable":
            pools[kind].append(f)

    rng = random.Random(args.seed)
    for pool in pools.values():
        rng.shuffle(pool)

    n_hsl_hold = max(1, int(len(pools["hsl"]) * args.holdout_hsl_frac))
    hold_hsl = pools["hsl"][:n_hsl_hold]
    train_hsl = pools["hsl"][n_hsl_hold:]

    hold_rgb = pools["rgbfunc"][: args.holdout_rgb]
    train_rgb = pools["rgbfunc"][args.holdout_rgb:
                                 args.holdout_rgb + args.train_rgb_quota]

    hold_other = pools["other"][: args.holdout_other]
    fill_budget = args.train_size - len(train_hsl) - len(train_rgb)
    if fill_budget < 0:
        raise SystemExit(
            f"--train-size {args.train_size} is smaller than the mandatory "
            f"hsl pool + rgb quota ({len(train_hsl)}+{len(train_rgb)}); "
            f"raise --train-size or lower --train-rgb-quota."
        )
    train_other = pools["other"][args.holdout_other:
                                 args.holdout_other + fill_budget]

    train = train_hsl + train_rgb + train_other
    holdout = hold_hsl + hold_rgb + hold_other
    rng.shuffle(train)   # file order inside the shard is immaterial but tidy

    overlap = set(map(str, train)) & set(map(str, holdout))
    assert not overlap, f"train/holdout overlap: {len(overlap)} files"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_files.txt").write_text(
        "\n".join(str(p) for p in train) + "\n", encoding="utf-8")
    (args.out_dir / "holdout_files.txt").write_text(
        "\n".join(str(p) for p in holdout) + "\n", encoding="utf-8")

    meta = {
        "seed": args.seed,
        "source_dirs": [str(args.css_dir)] + ([str(args.extra_dir)] if args.extra_dir else []),
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "train": {
            "total": len(train),
            "hsl_files": len(train_hsl),
            "rgbfunc_files": len(train_rgb),
            "other_files": len(train_other),
            "hsl_fraction": len(train_hsl) / max(1, len(train)),
        },
        "holdout": {
            "total": len(holdout),
            "hsl_files": len(hold_hsl),
            "rgbfunc_files": len(hold_rgb),
            "other_files": len(hold_other),
        },
    }
    (args.out_dir / "split_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"\nWrote {args.out_dir}/train_files.txt, holdout_files.txt")


if __name__ == "__main__":
    main()
