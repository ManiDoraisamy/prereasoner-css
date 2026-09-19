"""
Train N otherwise-identical models that differ only in random seed.

Everything else -- corpus, tokenizer, architecture, optimizer, schedule, step
count -- is held fixed. The seed drives both weight init and data order
(stage4_train calls torch.manual_seed before building the model and before
the shuffling DataLoader), so each run is an independent draw from the same
training distribution.

Checkpoints land in <out-root>/seed<N>/, which is what probes/seed_robustness.py
expects to be pointed at.

This is a thin sequential driver, not a scheduler: on CPU each 5000-step run
takes ~10h, so a 3-seed sweep is a multi-day job. Run it in the background and
check on it. Runs are independent -- if one dies, rerun that seed alone;
--skip-existing will leave finished seeds alone.

Usage
-----
    python pipeline/run_seeds.py --seeds 42 43 44 --steps 5000 --batch-size 4
    python pipeline/run_seeds.py --seeds 42 43 44 --steps 200 --smoke   # wiring check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).resolve().parent / "stage4_train.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--files", type=int, default=None,
                        help="Cap on training files (default: ALL files in the "
                             "pre-encoded shard — the stratified shard is "
                             "already the intended training set)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--lambda-sparse", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="data/parsed/stage4")
    parser.add_argument("--stage2-dir", default="data/parsed/stage2")
    parser.add_argument("--out-root", type=Path, default=Path("data/seeds"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a seed whose final checkpoint already exists.")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "manifest.json"
    runs = []

    for seed in args.seeds:
        out_dir = args.out_root / f"seed{seed}"
        final_ckpt = out_dir / f"step_{args.steps:06d}.pt"

        if args.skip_existing and final_ckpt.exists():
            print(f"\n=== seed {seed}: already done, skipping ===")
            runs.append({"seed": seed, "out_dir": str(out_dir),
                         "checkpoint": str(final_ckpt), "status": "skipped"})
            continue

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--seed", str(seed),
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            *(["--files", str(args.files)] if args.files else []),
            "--lr", str(args.lr),
            "--lambda-anchor", str(args.lambda_anchor),
            "--lambda-sparse", str(args.lambda_sparse),
            "--device", args.device,
            "--data-dir", args.data_dir,
            "--stage2-dir", args.stage2_dir,
            "--out-dir", str(out_dir),
        ]
        if args.smoke:
            cmd.append("--smoke")

        print(f"\n{'=' * 72}\n  seed {seed} -> {out_dir}\n{'=' * 72}")
        print("  " + " ".join(cmd))
        t0 = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0

        status = "ok" if proc.returncode == 0 else f"failed(rc={proc.returncode})"
        print(f"  seed {seed}: {status} in {elapsed / 3600:.2f} h")
        runs.append({
            "seed": seed,
            "out_dir": str(out_dir),
            "checkpoint": str(final_ckpt),
            "status": status,
            "elapsed_hours": round(elapsed / 3600, 3),
        })
        # Written after every seed so a crash mid-sweep still leaves a record.
        manifest = {
            "steps": args.steps, "batch_size": args.batch_size,
            "files": args.files, "lr": args.lr,
            "lambda_anchor": args.lambda_anchor,
            "lambda_sparse": args.lambda_sparse,
            "device": args.device, "smoke": args.smoke,
            "data_dir": args.data_dir, "runs": runs,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nManifest: {manifest_path}")
    ok = [r for r in runs if r["status"] in ("ok", "skipped")]
    print(f"{len(ok)}/{len(runs)} seeds available.")
    if len(ok) >= 2:
        ckpts = " ".join(f"--ckpt {r['checkpoint']}" for r in ok)
        print(f"\nNext:\n  python probes/seed_robustness.py {ckpts}")


if __name__ == "__main__":
    main()
