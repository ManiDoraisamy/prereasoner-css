"""
Stage 4 training: LM + RGB-anchor + sparsity constraint loss.

Two modes:
  --smoke   tiny model (hidden=64, 2 layers), 8 files, 20 steps, runs on CPU.
            Goal: verify gradients flow, losses decrease, checkpoint round-trips.
  default   full 38M model from Stage 3, configurable file count and steps,
            intended for A100 (CPU works but is impractically slow).

Usage:
  python stage4_train.py --smoke
  python stage4_train.py --files 1000 --steps 5000 --device cuda --wandb
"""

from __future__ import annotations

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import MmapWindowedDataset, collate_with_anchors
from loss import LossWeights, combined_loss, rgb_accuracy_stats
from model import NamedDimConfig, NamedDimModel
from tokenizer import CSSTokenizer

DATA_STAGE2 = Path("data/parsed/stage2")
# Checkpoint dir defaults to <--data-dir>/checkpoints at runtime inside train();
# --out-dir overrides it so multi-seed runs don't overwrite each other.


# ---------------------------------------------------------------------------
# LR schedule (linear warmup + cosine decay)
# ---------------------------------------------------------------------------

def cosine_schedule(total_steps: int, warmup: int):
    import math
    def lr_lambda(step: int):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"Device          : {device}")
    print(f"Mode            : {'SMOKE' if args.smoke else 'FULL'}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    stage2_dir = Path(args.stage2_dir) if args.stage2_dir else DATA_STAGE2
    print(f"\nLoading tokenizer from {stage2_dir}...")
    tok = CSSTokenizer.load(stage2_dir)
    print(f"  {tok}")

    # ------------------------------------------------------------------
    # Model config
    # ------------------------------------------------------------------
    if args.smoke:
        cfg = NamedDimConfig(
            vocab_size=tok.total_size,
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
            context_len=256,
        )
    else:
        cfg = NamedDimConfig(
            vocab_size=tok.total_size,
            hidden_dim=384,
            n_layers=6,
            n_heads=6,
            context_len=1024,
        )

    model = NamedDimModel(cfg).to(device)
    print(f"\nModel params    : {model.num_parameters() / 1e6:.2f}M")
    print(f"Hidden dim      : {cfg.hidden_dim}")
    print(f"Context length  : {cfg.context_len}")
    print(f"Named dims      : {model.named_dims}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    # Checkpoint dir is derived from --data-dir so smoke runs write to their
    # own data dir's checkpoints, not the real production tree. --out-dir
    # overrides it, which is how multi-seed runs keep their checkpoints apart.
    ckpt_dir = Path(args.out_dir) if args.out_dir else data_dir / "checkpoints"
    if not (data_dir / "tokens.bin").exists():
        print(f"\nERROR: pre-encoded blob not found at {data_dir}/tokens.bin")
        print("Run:  python stage4_preencode.py  [--limit N]  [--out DIR]")
        return
    print(f"\nLoading mmap dataset from {data_dir}/")
    dataset = MmapWindowedDataset(
        data_dir,
        context_len=cfg.context_len,
        max_files=args.files,
    )
    print(f"  {len(dataset):,} training windows from up to {args.files or 'all'} files")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_with_anchors(b, pad_id=tok.pad_id or 0),
        num_workers=0,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Optimizer / scheduler / loss config
    # ------------------------------------------------------------------
    optim = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = LambdaLR(optim, cosine_schedule(args.steps, warmup=max(10, args.steps // 20)))
    weights = LossWeights(lambda_anchor=args.lambda_anchor, lambda_sparse=args.lambda_sparse)

    print(f"\nOptim           : AdamW lr={args.lr}, wd=0.1")
    print(f"Steps           : {args.steps}")
    print(f"Batch size      : {args.batch_size}")
    print(f"lambda_anchor   : {weights.lambda_anchor}")
    print(f"lambda_sparse   : {weights.lambda_sparse}")

    # ------------------------------------------------------------------
    # W&B (optional)
    # ------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="prereasoner-stage4",
            config={
                "smoke": args.smoke,
                "hidden_dim": cfg.hidden_dim,
                "n_layers": cfg.n_layers,
                "context_len": cfg.context_len,
                "files": len(dataset),
                "batch_size": args.batch_size,
                "steps": args.steps,
                "lr": args.lr,
                "lambda_anchor": weights.lambda_anchor,
                "lambda_sparse": weights.lambda_sparse,
                "params_M": model.num_parameters() / 1e6,
            },
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    data_iter = iter(loader)
    t0 = time.time()
    log_every = max(1, args.steps // 20)
    ckpt_every = (args.ckpt_every if args.ckpt_every is not None
                  else max(log_every, args.steps // 5))

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        anchor_batch = batch["anchor_batch"].to(device)
        anchor_pos = batch["anchor_pos"].to(device)
        anchor_rgb = batch["anchor_rgb"].to(device)
        anchor_channel = batch["anchor_channel"].to(device)

        # Labels = input_ids with pad positions masked to -100
        labels = input_ids.clone()
        labels[attn == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        hidden = out.hidden_states[-1]

        losses = combined_loss(
            out, hidden, anchor_batch, anchor_pos, anchor_rgb, anchor_channel, weights,
        )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        step += 1

        if step % log_every == 0 or step == 1 or step == args.steps:
            stats = rgb_accuracy_stats(
                hidden, anchor_batch, anchor_pos, anchor_rgb, anchor_channel,
            )
            elapsed = time.time() - t0
            rate = step / elapsed
            print(
                f"step {step:>5d}  "
                f"lm={losses.lm_loss.item():.3f}  "
                f"anc={losses.anchor_loss.item():.4f}  "
                f"sp={losses.sparse_loss.item():.4f}  "
                f"tot={losses.total.item():.3f}  "
                f"n_anc={losses.n_anchors:>4d}  "
                f"rgb_mae={stats['rgb_mae']:.3f}  "
                f"lr={sched.get_last_lr()[0]:.2e}  "
                f"{rate:.2f} step/s"
            )
            if wandb_run is not None:
                wandb_run.log({
                    "step": step,
                    "loss/lm": losses.lm_loss.item(),
                    "loss/anchor": losses.anchor_loss.item(),
                    "loss/sparse": losses.sparse_loss.item(),
                    "loss/total": losses.total.item(),
                    "rgb/mae": stats["rgb_mae"],
                    "rgb/max_err": stats["rgb_max_err"],
                    "rgb/corr_r": stats["rgb_corr_r"],
                    "rgb/corr_g": stats["rgb_corr_g"],
                    "rgb/corr_b": stats["rgb_corr_b"],
                    "n_anchors": losses.n_anchors,
                    "lr": sched.get_last_lr()[0],
                })

        if step % ckpt_every == 0 or step == args.steps:
            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "sched_state": sched.state_dict(),
                "config": {
                    "hidden_dim": cfg.hidden_dim,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "context_len": cfg.context_len,
                    "vocab_size": cfg.vocab_size,
                    "lambda_anchor": weights.lambda_anchor,
                    "lambda_sparse": weights.lambda_sparse,
                },
            }, ckpt_path)
            print(f"  -> checkpoint: {ckpt_path}  ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s ({args.steps / elapsed:.2f} step/s avg).")
    print(f"Checkpoints saved to {ckpt_dir}")
    if wandb_run is not None:
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny model + few steps to verify the loop works")
    parser.add_argument("--files", type=int, default=None,
                        help="Number of token streams to train on (default: all)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--lambda-sparse", type=float, default=0.1)
    parser.add_argument("--device", default=None,
                        help="cuda|cpu (auto-detect if not given)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--out-dir", default=None,
                        help="Checkpoint output dir (default: <data-dir>/checkpoints). "
                             "Set per seed so multi-seed runs don't collide.")
    parser.add_argument("--stage2-dir", default=None,
                        help=f"Tokenizer dir (default: {DATA_STAGE2})")
    parser.add_argument("--data-dir", default="data/parsed/stage4",
                        help="Directory with the pre-encoded blob "
                             "(tokens.bin, file_offsets.bin, anchors.bin, anchor_offsets.bin, meta.json)")
    parser.add_argument("--ckpt-every", type=int, default=None,
                        help="Save checkpoint every N steps (default: steps//5)")
    args = parser.parse_args()

    # Smoke-mode defaults
    if args.smoke:
        if args.files is None: args.files = 8
        if args.steps is None: args.steps = 20
        if args.batch_size is None: args.batch_size = 2
    else:
        if args.files is None: args.files = 1000
        if args.steps is None: args.steps = 5000
        if args.batch_size is None: args.batch_size = 8

    # Device auto-detect
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train(args)


if __name__ == "__main__":
    main()
