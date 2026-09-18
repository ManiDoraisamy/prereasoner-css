"""
Stage 8: Discover named dimensions that emerged WITHOUT supervision.

The model was trained with only R, G, B anchored. If the methodology works,
additional dimensions should have spontaneously specialized to encode CSS
color concepts the model needed for prediction — most cleanly, the
H (hue), S (saturation), L (lightness) of HSL colors that appear in CSS.

This script:
  1. Extracts HSL color expressions from the corpus (real, naturally-occurring
     examples — not synthetic).
  2. Runs the trained model on each, capturing the final-layer hidden state
     at the closing ')' of each hsl(...) call.
  3. For each of the 381 non-RGB dimensions, computes correlation with H, S,
     L across the example set. Hue is treated circularly (sin & cos basis).
  4. Identifies candidate dimensions (correlation > THRESHOLD with some
     HSL property).
  5. Verifies on a held-out split: refit a univariate linear predictor on
     train activations, score it on test activations.
  6. Also fits a multivariate linear predictor (uses all 381 dims) to check
     overall predictability and detect entanglement.

Usage:
  python stage8_discover.py
  python stage8_discover.py --n-files 5000 --max-examples 3000
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
import torch
from tqdm import tqdm

from color_detection import detect_colors
from css_walker import walk_css
from model import NamedDimConfig, NamedDimModel
from tokenizer import CSSTokenizer

DEFAULT_CKPT = Path("data/parsed/stage4/checkpoints/step_005000.pt")
OUT_DIR = Path("data/parsed/stage8")


# ---------------------------------------------------------------------------
# Stage 1: extract HSL examples
# ---------------------------------------------------------------------------

def extract_hsl_examples(tok: CSSTokenizer, n_files: int, max_examples: int,
                        context_len: int):
    """Walk CSS files, find every parseable hsl()/hsla(), return a list of
    {ids, pos, h, s, l, source_file}."""
    files = sorted(Path("data/stage1/css").glob("*.css"))[:n_files]
    examples = []
    skipped_unparseable = 0

    for path in tqdm(files, desc="Extracting HSL"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            stream = walk_css(text)
        except Exception:
            continue
        _, hsls = detect_colors(stream)
        if not hsls:
            continue

        ids, stream_to_id = tok.encode_tokens_with_positions(
            stream, add_special=True,
        )

        for h_det in hsls:
            if h_det.h is None or h_det.s is None or h_det.l is None:
                skipped_unparseable += 1
                continue
            id_pos = stream_to_id[h_det.position]

            # Build a window that ends at id_pos and is at most context_len long
            if id_pos >= context_len:
                start = id_pos - context_len + 1
                window_ids = ids[start: id_pos + 1]
                window_pos = id_pos - start
            else:
                # Keep tokens [0 .. min(id_pos + lookahead, len(ids))]
                end = min(len(ids), context_len)
                window_ids = ids[:end]
                window_pos = id_pos

            examples.append({
                "ids": window_ids,
                "pos": int(window_pos),
                "h": float(h_det.h),
                "s": float(h_det.s),
                "l": float(h_det.l),
                "source_file": path.name,
            })
            if len(examples) >= max_examples:
                return examples, skipped_unparseable

    return examples, skipped_unparseable


# ---------------------------------------------------------------------------
# Stage 2: run inference, capture hidden states at HSL positions
# ---------------------------------------------------------------------------

def run_inference(model, examples, hidden_dim: int, device: str):
    activations = np.zeros((len(examples), hidden_dim), dtype=np.float32)
    pad_id = 0
    for i, ex in enumerate(tqdm(examples, desc="Inferring")):
        ids = torch.tensor([ex["ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=ids)
        hidden = out.hidden_states[-1][0, ex["pos"]].cpu().numpy()
        activations[i] = hidden
    return activations


# ---------------------------------------------------------------------------
# Stage 3: per-dimension correlation analysis
# ---------------------------------------------------------------------------

def safe_corr(a, b):
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def per_dim_correlations(activations, h_vals, s_vals, l_vals,
                         named_dims=(0, 1, 2)):
    """For each non-RGB dim, return its max correlation with H (sin or cos
    basis), S, and L. Returns list of dicts sorted by best_r descending."""
    h = np.asarray(h_vals, dtype=np.float64)
    s = np.asarray(s_vals, dtype=np.float64)
    l = np.asarray(l_vals, dtype=np.float64)

    # Hue is circular: encode as sin/cos for correlation
    h_rad = h * np.pi / 180.0
    h_sin = np.sin(h_rad)
    h_cos = np.cos(h_rad)

    n_dims = activations.shape[1]
    results = []
    for d in range(n_dims):
        if d in named_dims:
            continue
        col = activations[:, d].astype(np.float64)
        r_h_sin = abs(safe_corr(col, h_sin))
        r_h_cos = abs(safe_corr(col, h_cos))
        r_h = max(r_h_sin, r_h_cos)
        r_s = abs(safe_corr(col, s))
        r_l = abs(safe_corr(col, l))
        best_prop, best_r = max(
            [("H", r_h), ("S", r_s), ("L", r_l)],
            key=lambda x: x[1],
        )
        results.append({
            "dim": d,
            "r_h": r_h,
            "r_s": r_s,
            "r_l": r_l,
            "best_prop": best_prop,
            "best_r": best_r,
        })
    return sorted(results, key=lambda x: -x["best_r"])


# ---------------------------------------------------------------------------
# Stage 4 / 5: held-out verification
# ---------------------------------------------------------------------------

def linear_fit_score(x_train, y_train, x_test, y_test):
    """Univariate linear regression; return test R^2."""
    if x_train.std() < 1e-6 or y_train.std() < 1e-6:
        return None
    A = np.stack([x_train, np.ones_like(x_train)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    pred = x_test * coef[0] + coef[1]
    ss_res = ((y_test - pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    if ss_tot < 1e-9:
        return None
    return 1.0 - ss_res / ss_tot


def multivariate_r2(X_train, y_train, X_test, y_test, ridge=1e-3):
    """Multivariate ridge regression; return test R^2."""
    if y_train.std() < 1e-6:
        return None
    # Add intercept
    A = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    AtA = A.T @ A
    AtA[:-1, :-1] += ridge * np.eye(A.shape[1] - 1)
    coef = np.linalg.solve(AtA, A.T @ y_train)
    pred = np.hstack([X_test, np.ones((X_test.shape[0], 1))]) @ coef
    ss_res = ((y_test - pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-files", type=int, default=5000,
                        help="How many CSS files to scan for HSL examples")
    parser.add_argument("--max-examples", type=int, default=2000,
                        help="Cap on the number of HSL examples to use")
    parser.add_argument("--candidate-threshold", type=float, default=0.5,
                        help="Min |correlation| to flag a candidate dim")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction of examples held out for verification")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = CSSTokenizer.load("data/parsed/stage2")

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, weights_only=False, map_location=args.device)
    cfg_dict = ckpt["config"]
    cfg = NamedDimConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_dim=cfg_dict["hidden_dim"],
        n_layers=cfg_dict["n_layers"],
        n_heads=cfg_dict["n_heads"],
        context_len=cfg_dict["context_len"],
    )
    model = NamedDimModel(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  hidden_dim={cfg.hidden_dim}  layers={cfg.n_layers}  step={ckpt['step']}")

    # 1. Extract HSL examples
    print(f"\n[1] Extracting HSL examples from up to {args.n_files} files...")
    examples, skipped = extract_hsl_examples(
        tok, n_files=args.n_files, max_examples=args.max_examples,
        context_len=cfg.context_len,
    )
    print(f"  Collected: {len(examples)}")
    print(f"  Skipped (unparseable H/S/L): {skipped}")
    if len(examples) < 100:
        print("ERROR: too few HSL examples in corpus. Increase --n-files.")
        return

    # 2. Inference
    print(f"\n[2] Running inference on {len(examples)} HSL positions...")
    activations = run_inference(model, examples, cfg.hidden_dim, args.device)

    h_vals = np.array([e["h"] for e in examples])
    s_vals = np.array([e["s"] for e in examples])
    l_vals = np.array([e["l"] for e in examples])

    print(f"\n  Value ranges:")
    print(f"    H : {h_vals.min():6.1f} .. {h_vals.max():6.1f}   "
          f"mean={h_vals.mean():.1f}  std={h_vals.std():.1f}")
    print(f"    S : {s_vals.min():6.1f} .. {s_vals.max():6.1f}   "
          f"mean={s_vals.mean():.1f}  std={s_vals.std():.1f}")
    print(f"    L : {l_vals.min():6.1f} .. {l_vals.max():6.1f}   "
          f"mean={l_vals.mean():.1f}  std={l_vals.std():.1f}")

    # 3. Per-dim correlations (on full set)
    print(f"\n[3] Per-dim correlation analysis (across {cfg.hidden_dim - 3} non-RGB dims)...")
    corrs = per_dim_correlations(activations, h_vals, s_vals, l_vals)

    print(f"\n  Top 20 dims by best |r| (across H, S, L):")
    print(f"  {'rank':<5}{'dim':<6}{'best':<6}{'best_r':<8}{'r_H':<8}{'r_S':<8}{'r_L':<8}")
    for i, r in enumerate(corrs[:20]):
        print(f"  {i+1:<5}{r['dim']:<6}{r['best_prop']:<6}"
              f"{r['best_r']:<8.3f}{r['r_h']:<8.3f}{r['r_s']:<8.3f}{r['r_l']:<8.3f}")

    # Candidates per property
    candidates = {"H": [], "S": [], "L": []}
    for r in corrs:
        if r["best_r"] >= args.candidate_threshold:
            candidates[r["best_prop"]].append(r)
    print(f"\n  Candidate dims (|r| >= {args.candidate_threshold}):")
    for prop in ("H", "S", "L"):
        n = len(candidates[prop])
        print(f"    {prop}: {n} dim(s)")
        for r in candidates[prop][:5]:
            print(f"      dim {r['dim']}  |r_{prop}|={r['best_r']:.3f}")

    # 4. Held-out verification
    print(f"\n[4] Held-out verification (test split = {args.test_split:.0%})...")
    n = len(examples)
    test_size = max(50, int(n * args.test_split))
    perm = np.random.permutation(n)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]

    print(f"  Train: {len(train_idx)}   Test: {len(test_idx)}")

    # Re-correlate on train, evaluate on test for the top candidates
    h_train_sin = np.sin(h_vals[train_idx] * np.pi / 180)
    h_train_cos = np.cos(h_vals[train_idx] * np.pi / 180)
    h_test_sin = np.sin(h_vals[test_idx] * np.pi / 180)
    h_test_cos = np.cos(h_vals[test_idx] * np.pi / 180)

    print(f"\n  Univariate linear fit on TRAIN, R^2 on TEST:")
    print(f"  {'dim':<6}{'prop':<6}{'train_|r|':<11}{'test_R^2':<10}")
    verified = []
    for prop in ("H", "S", "L"):
        for r in candidates[prop][:5]:
            d = r["dim"]
            x_train = activations[train_idx, d]
            x_test = activations[test_idx, d]
            if prop == "H":
                # Try both sin and cos bases, keep the better one
                r2_sin = linear_fit_score(x_train, h_train_sin, x_test, h_test_sin)
                r2_cos = linear_fit_score(x_train, h_train_cos, x_test, h_test_cos)
                test_r2 = max(filter(lambda v: v is not None, [r2_sin or -1, r2_cos or -1]),
                              default=None)
            elif prop == "S":
                test_r2 = linear_fit_score(x_train, s_vals[train_idx], x_test, s_vals[test_idx])
            else:
                test_r2 = linear_fit_score(x_train, l_vals[train_idx], x_test, l_vals[test_idx])
            if test_r2 is None:
                continue
            verified.append({"dim": d, "prop": prop, "train_r": r["best_r"], "test_r2": test_r2})
            print(f"  {d:<6}{prop:<6}{r['best_r']:<11.3f}{test_r2:<10.3f}")

    # 5. Multivariate predictability (all 381 non-RGB dims)
    print(f"\n[5] Multivariate ridge regression (all 381 non-RGB dims):")
    non_rgb = np.array([d for d in range(cfg.hidden_dim) if d not in (0, 1, 2)])
    X = activations[:, non_rgb]
    X_train, X_test = X[train_idx], X[test_idx]

    # H: use 2D target (sin, cos), report combined R^2
    H_train = np.stack([h_train_sin, h_train_cos], axis=1)
    H_test = np.stack([h_test_sin, h_test_cos], axis=1)
    # Predict each separately
    r2_h_sin = multivariate_r2(X_train, H_train[:, 0], X_test, H_test[:, 0])
    r2_h_cos = multivariate_r2(X_train, H_train[:, 1], X_test, H_test[:, 1])
    r2_s = multivariate_r2(X_train, s_vals[train_idx], X_test, s_vals[test_idx])
    r2_l = multivariate_r2(X_train, l_vals[train_idx], X_test, l_vals[test_idx])

    print(f"  Test R^2 predicting:")
    print(f"    sin(H)  : {r2_h_sin:.3f}" if r2_h_sin is not None else "    sin(H)  : N/A")
    print(f"    cos(H)  : {r2_h_cos:.3f}" if r2_h_cos is not None else "    cos(H)  : N/A")
    print(f"    S       : {r2_s:.3f}" if r2_s is not None else "    S       : N/A")
    print(f"    L       : {r2_l:.3f}" if r2_l is not None else "    L       : N/A")

    # 6. Save results
    results_path = OUT_DIR / "discovery_results.json"
    out = {
        "checkpoint": str(args.ckpt),
        "step": ckpt["step"],
        "hidden_dim": cfg.hidden_dim,
        "n_examples": len(examples),
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "candidates": {
            prop: [{"dim": r["dim"], "r": r["best_r"]} for r in candidates[prop]]
            for prop in ("H", "S", "L")
        },
        "verified": verified,
        "multivariate_test_r2": {
            "sin_H": r2_h_sin,
            "cos_H": r2_h_cos,
            "S": r2_s,
            "L": r2_l,
        },
        "top_50_per_dim": [
            {"dim": r["dim"], "best_prop": r["best_prop"],
             "best_r": r["best_r"], "r_H": r["r_h"], "r_S": r["r_s"], "r_L": r["r_l"]}
            for r in corrs[:50]
        ],
    }
    with results_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Headline
    print(f"\n{'=' * 60}")
    print(f"  Stage 8 verdict")
    print(f"{'=' * 60}")
    n_h = len(candidates["H"])
    n_s = len(candidates["S"])
    n_l = len(candidates["L"])
    n_total = n_h + n_s + n_l
    print(f"  Candidate dims (|r| >= {args.candidate_threshold}) : "
          f"{n_total}  (H:{n_h}  S:{n_s}  L:{n_l})")
    if verified:
        passed = [v for v in verified if v["test_r2"] is not None and v["test_r2"] > 0.3]
        print(f"  Held-out verified (test R^2 > 0.3)            : {len(passed)}")
    print(f"  Multivariate-recoverable H, S, L predictability:")
    for label, r2 in [("sin(H)", r2_h_sin), ("cos(H)", r2_h_cos),
                      ("S", r2_s), ("L", r2_l)]:
        flag = "" if r2 is None else (" GOOD" if r2 > 0.5 else " WEAK" if r2 > 0.2 else " NONE")
        print(f"    {label:<8}: R^2={r2 if r2 is not None else float('nan'):.3f}{flag}")


if __name__ == "__main__":
    main()
