"""
How much of a concept is recoverable — from one dimension, vs from all of them?

Answers a specific question: anchored RGB reads off a single dimension at
r ~ 0.998. Can HSL match that if you stop restricting yourself to one
dimension and fit a probe over the whole hidden state?

If pooling all 381 unanchored dims still falls well short of anchored RGB,
then anchoring buys PRECISION, not merely a known location — the model
simply never encoded hue that accurately, because next-token prediction
never required it.

Reports, per property, on held-out examples:
  - best single dim (selected on half A, scored on half B)
  - ridge over ALL non-anchored dims (fit on half A, scored on half B)
  - for reference, anchored RGB: direct read of dim 0/1/2, no fitting

Usage:
    python probes/capacity_compare.py \
        --ckpt data/seeds/seed42/step_005000.pt --label seed42 \
        --ckpt data/seeds/baseline42/step_005000.pt --label baseline \
        --file-list data/splits/holdout_files.txt
"""

from __future__ import annotations

# --- repo path shim ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import NamedDimConfig, NamedDimModel
from tokenizer import CSSTokenizer
from stage8c_multiprobe import extract_multiprobe_examples, safe_corr
from seed_robustness import (
    CANONICAL_PROBE, collect_activations, property_targets,
    all_dim_correlations_fast,
)


def ridge_r2(X_tr, y_tr, X_te, y_te, alpha=1.0):
    """Closed-form ridge, then R^2 on held-out half. Standardized features."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    Xtr = (X_tr - mu) / sd
    Xte = (X_te - mu) / sd
    ym = y_tr.mean()
    yc = y_tr - ym
    d = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ yc)
    pred = Xte @ w + ym
    ss_res = float(((y_te - pred) ** 2).sum())
    ss_tot = float(((y_te - y_te.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, action="append", required=True)
    p.add_argument("--label", action="append", default=None)
    p.add_argument("--stage2-dir", default="data/parsed/stage2")
    p.add_argument("--file-list", type=Path,
                   default=Path("data/splits/holdout_files.txt"))
    p.add_argument("--n-files", type=int, default=8439)
    p.add_argument("--max-examples", type=int, default=1200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path,
                   default=Path("results/capacity_compare.json"))
    args = p.parse_args()

    labels = args.label or [c.parent.name for c in args.ckpt]
    tok = CSSTokenizer.load(args.stage2_dir)

    ck = torch.load(args.ckpt[0], weights_only=False, map_location=args.device)
    ctx = ck["config"]["context_len"]
    print("Extracting HSL examples...")
    examples = extract_multiprobe_examples(
        tok, args.n_files, args.max_examples, ctx,
        file_list=args.file_list,
    )
    print(f"  {len(examples)} examples")
    if len(examples) < 200:
        raise SystemExit("not enough examples")

    n = len(examples)
    perm = np.random.default_rng(1).permutation(n)
    ia, ib = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    tgts = property_targets(examples)

    out = {}
    for path, lbl in zip(args.ckpt, labels):
        print(f"\n=== {lbl} ===")
        c = torch.load(path, weights_only=False, map_location=args.device)
        cfg = NamedDimConfig(**{k: c["config"][k] for k in
                                ("vocab_size", "hidden_dim", "n_layers",
                                 "n_heads", "context_len")})
        model = NamedDimModel(cfg).to(args.device)
        model.load_state_dict(c["model_state"])
        model.eval()
        acts = collect_activations(model, examples, cfg.hidden_dim, args.device)

        rows = {}
        print(f"  {'prop':<5}{'best dim':>10}{'1-dim r2':>11}{'all-dim R2':>12}"
              f"{'gain':>8}")
        for prop, probe in CANONICAL_PROBE.items():
            a = acts[probe]
            if prop == "H":
                rs = all_dim_correlations_fast(a[ia], tgts["H_sin"][ia])
                rc = all_dim_correlations_fast(a[ia], tgts["H_cos"][ia])
                y = tgts["H_sin"] if rs.max() >= rc.max() else tgts["H_cos"]
            else:
                y = tgts[prop]

            vec = all_dim_correlations_fast(a[ia], y[ia])
            best = int(np.argmax(vec))
            one_r = abs(safe_corr(a[ib, best], y[ib]))
            one_r2 = one_r ** 2

            # All dims EXCEPT the supervised 0/1/2 — the honest "everything
            # the model organized on its own" pool.
            cols = [d for d in range(cfg.hidden_dim) if d not in (0, 1, 2)]
            all_r2 = ridge_r2(a[np.ix_(ia, cols)], y[ia],
                              a[np.ix_(ib, cols)], y[ib])
            rows[prop] = {"best_dim": best, "one_dim_r": one_r,
                          "one_dim_r2": one_r2, "all_dim_r2": all_r2,
                          "gain": all_r2 - one_r2}
            print(f"  {prop:<5}{best:>10}{one_r2:>11.3f}{all_r2:>12.3f}"
                  f"{all_r2 - one_r2:>+8.3f}")
        out[lbl] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n_examples": n, "results": out}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print("\nCompare against anchored RGB direct read: r=0.998 -> r2=0.996,")
    print("from ONE dimension with no probe fitted.")


if __name__ == "__main__":
    main()
