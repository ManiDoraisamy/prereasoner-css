"""
Unattended driver for the full pre-registered experiment (docs/PROTOCOL.md).

Runs, in order:
  1. hsl enrichment      (optional, time-budgeted)
  2. stratified splits   (train shard + holdout)
  3. token extraction    (TRAIN SHARD ONLY -- holdout never sees the vocab)
  4. vocabulary build
  5. tokenizer round-trip check   [gate: must pass]
  6. pre-encode train shard
  7. train 3 anchored seeds (42, 43, 44)
  8. train 1 LM-only baseline (lambda_anchor=0, lambda_sparse=0)

Each step is skipped when its output artifact already exists, so a crash or
a kill can be resumed by re-running the same command. --force-from <step>
re-runs from that step onward.

Total on CPU: ~45 h (~2 h data, ~40 h training, enrichment on top).

Usage:
    python run_experiment.py                      # full sequence
    python run_experiment.py --no-enrich          # skip step 1
    python run_experiment.py --force-from 3       # rebuild vocab onward
    python run_experiment.py --dry-run            # print the plan only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent
LOG = ROOT / "data" / "experiment_log.jsonl"

SEEDS = [42, 43, 44]
STEPS = 5000
BATCH = 4


def log_event(event: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def run(cmd: list[str], label: str, dry: bool) -> None:
    print(f"\n{'=' * 74}\n  {label}\n{'=' * 74}")
    print("  " + " ".join(str(c) for c in cmd))
    if dry:
        return
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    elapsed = time.time() - t0
    log_event({"step": label, "returncode": proc.returncode,
               "elapsed_hours": round(elapsed / 3600, 4)})
    if proc.returncode != 0:
        raise SystemExit(f"\nFAILED at: {label} (rc={proc.returncode}).\n"
                         f"Fix, then re-run — completed steps are skipped.")
    print(f"  done in {elapsed / 60:.1f} min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-enrich", action="store_true")
    p.add_argument("--enrich-target", type=int, default=3000)
    p.add_argument("--enrich-minutes", type=float, default=150)
    p.add_argument("--force-from", type=int, default=None,
                   help="Re-run from this step number (1-8) onward")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-training", action="store_true",
                   help="Stop after pre-encoding (steps 1-6)")
    args = p.parse_args()

    dry = args.dry_run
    ff = args.force_from or 99

    def should(step: int, artifact: Path | None) -> bool:
        if step >= ff:
            return True
        if artifact is None:
            return True
        if artifact.exists():
            print(f"\n[step {step}] artifact exists, skipping: {artifact}")
            return False
        return True

    enrich_dir = ROOT / "data" / "stage1_enrich" / "css"
    splits = ROOT / "data" / "splits" / "train_files.txt"
    holdout = ROOT / "data" / "splits" / "holdout_files.txt"
    freq = ROOT / "data" / "parsed" / "stage2" / "token_freq.json"
    fathead = ROOT / "data" / "parsed" / "stage2" / "fat_head.json"
    tokens = ROOT / "data" / "parsed" / "stage4" / "tokens.bin"

    # ---- 1. enrichment -------------------------------------------------
    if not args.no_enrich and should(1, None):
        run([PY, "pipeline/stage1_enrich_hsl.py",
             "--target", args.enrich_target,
             "--max-minutes", args.enrich_minutes],
            "step 1/8  hsl enrichment", dry)

    # ---- 2. splits -----------------------------------------------------
    if should(2, splits):
        cmd = [PY, "pipeline/make_splits.py", "--css-dir", "data/stage1/css"]
        if enrich_dir.exists() and any(enrich_dir.glob("*.css")):
            cmd += ["--extra-dir", str(enrich_dir.relative_to(ROOT))]
        run(cmd, "step 2/8  stratified splits", dry)

    # ---- 3. token extraction (train shard only) ------------------------
    if should(3, freq):
        run([PY, "pipeline/stage2_extract_tokens.py",
             "--file-list", "data/splits/train_files.txt"],
            "step 3/8  token extraction (train shard only)", dry)

    # ---- 4. vocabulary -------------------------------------------------
    if should(4, fathead):
        run([PY, "pipeline/stage2_build_vocab.py"],
            "step 4/8  vocabulary build", dry)

    # ---- 5. round-trip gate --------------------------------------------
    if should(5, None):
        run([PY, "pipeline/stage2_test_roundtrip.py", "--sample", 1000],
            "step 5/8  tokenizer round-trip gate", dry)

    # ---- 6. pre-encode --------------------------------------------------
    if should(6, tokens):
        run([PY, "pipeline/stage4_preencode.py",
             "--file-list", "data/splits/train_files.txt",
             "--out", "data/parsed/stage4"],
            "step 6/8  pre-encode train shard", dry)

    if args.skip_training:
        print("\n--skip-training set; stopping after pre-encode.")
        return

    # ---- 7. anchored seeds ----------------------------------------------
    anchored_done = all(
        (ROOT / "data" / "seeds" / f"seed{s}" / f"step_{STEPS:06d}.pt").exists()
        for s in SEEDS
    )
    if should(7, None) and not (anchored_done and ff > 7):
        run([PY, "pipeline/run_seeds.py",
             "--seeds", *[str(s) for s in SEEDS],
             "--steps", STEPS, "--batch-size", BATCH,
             "--data-dir", "data/parsed/stage4",
             "--stage2-dir", "data/parsed/stage2",
             "--out-root", "data/seeds",
             "--skip-existing"],
            f"step 7/8  train anchored seeds {SEEDS}", dry)

    # ---- 8. LM-only baseline --------------------------------------------
    baseline_ckpt = ROOT / "data" / "seeds" / "baseline42" / f"step_{STEPS:06d}.pt"
    if should(8, baseline_ckpt):
        run([PY, "pipeline/stage4_train.py",
             "--seed", 42, "--steps", STEPS, "--batch-size", BATCH,
             "--lambda-anchor", 0.0, "--lambda-sparse", 0.0,
             "--data-dir", "data/parsed/stage4",
             "--stage2-dir", "data/parsed/stage2",
             "--out-dir", "data/seeds/baseline42"],
            "step 8/8  LM-only baseline (lambda=0)", dry)

    print(f"\n{'=' * 74}\n  ALL STEPS COMPLETE\n{'=' * 74}")
    print("Checkpoints:")
    for s in SEEDS:
        print(f"  data/seeds/seed{s}/step_{STEPS:06d}.pt")
    print(f"  data/seeds/baseline42/step_{STEPS:06d}.pt   (lambda=0 control)")
    print("\nNext — the pre-registered analyses (docs/PROTOCOL.md):")
    ck = " ".join(f"--ckpt data/seeds/seed{s}/step_{STEPS:06d}.pt" for s in SEEDS)
    print(f"  python probes/seed_robustness.py {ck} \\\n"
          f"      --file-list data/splits/holdout_files.txt \\\n"
          f"      --out results/seed_robustness.json")


if __name__ == "__main__":
    main()
