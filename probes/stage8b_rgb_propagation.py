"""
Stage 8b: Test RGB propagation to HSL positions.

The named R, G, B dimensions (indices 0, 1, 2) were anchored ONLY at hex,
named-color, and rgb() positions during training. The model was never given
HSL supervision.

But the README's Phase 0 spec predicts that R, G, B should still activate
at HSL positions with the correct values — because predicting the tokens
that follow an HSL color (in real CSS, often more colors or transparency
adjustments) requires the model to "know" what color the HSL refers to.

This script:
  1. Extracts HSL color expressions from the corpus.
  2. Converts each (H, S, L) to its standard (R, G, B) equivalent.
  3. Runs the model on each example, captures hidden state at the closing `)`.
  4. Checks whether dims 0, 1, 2 correlate with the TRUE RGB the HSL encodes.

If correlations are high, the constraint mechanism didn't just memorize —
it propagated the named concept to a related but unsupervised color format.
That's a substantial positive result.
"""

from __future__ import annotations

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import colorsys
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


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    """Standard HSL→RGB. h in [0,360), s,l in [0,100]. Returns r,g,b in [0,1]."""
    return colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)


def extract_hsl_examples(tok, n_files, max_examples, context_len):
    """Walk CSS files, find every parseable hsl()/hsla(), return list of dicts
    with the model input window plus the position and the TRUE RGB."""
    files = sorted(Path("data/stage1/css").glob("*.css"))[:n_files]
    examples = []
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
                continue
            id_pos = stream_to_id[h_det.position]
            if id_pos >= context_len:
                start = id_pos - context_len + 1
                window_ids = ids[start: id_pos + 1]
                window_pos = id_pos - start
            else:
                end = min(len(ids), context_len)
                window_ids = ids[:end]
                window_pos = id_pos
            r, g, b = hsl_to_rgb(h_det.h, h_det.s, h_det.l)
            examples.append({
                "ids": window_ids,
                "pos": int(window_pos),
                "h": float(h_det.h),
                "s": float(h_det.s),
                "l": float(h_det.l),
                "r_true": float(r),
                "g_true": float(g),
                "b_true": float(b),
            })
            if len(examples) >= max_examples:
                return examples
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-files", type=int, default=20000)
    parser.add_argument("--max-examples", type=int, default=2000)
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = CSSTokenizer.load("data/parsed/stage2")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, weights_only=False, map_location=args.device)
    cfg = NamedDimConfig(**{k: ckpt["config"][k] for k in
                            ("vocab_size", "hidden_dim", "n_layers", "n_heads", "context_len")})
    model = NamedDimModel(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  step={ckpt['step']}  hidden={cfg.hidden_dim}")

    print(f"\n[1] Extracting HSL examples (up to {args.max_examples})...")
    examples = extract_hsl_examples(
        tok, args.n_files, args.max_examples, cfg.context_len,
    )
    print(f"  Collected: {len(examples)}")

    print(f"\n[2] Running inference, capturing dims 0/1/2 at HSL positions...")
    r_pred = np.zeros(len(examples), dtype=np.float32)
    g_pred = np.zeros(len(examples), dtype=np.float32)
    b_pred = np.zeros(len(examples), dtype=np.float32)
    r_true = np.array([e["r_true"] for e in examples])
    g_true = np.array([e["g_true"] for e in examples])
    b_true = np.array([e["b_true"] for e in examples])

    for i, ex in enumerate(tqdm(examples, desc="Inferring")):
        ids = torch.tensor([ex["ids"]], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model(input_ids=ids)
        hidden = out.hidden_states[-1][0, ex["pos"]].cpu().numpy()
        r_pred[i] = hidden[0]
        g_pred[i] = hidden[1]
        b_pred[i] = hidden[2]

    # -------------------------------------------------------------------
    print(f"\n[3] Correlation: R, G, B named dims vs TRUE RGB derived from HSL")
    corr_r = float(np.corrcoef(r_pred, r_true)[0, 1])
    corr_g = float(np.corrcoef(g_pred, g_true)[0, 1])
    corr_b = float(np.corrcoef(b_pred, b_true)[0, 1])

    mae_r = float(np.abs(r_pred - r_true).mean())
    mae_g = float(np.abs(g_pred - g_true).mean())
    mae_b = float(np.abs(b_pred - b_true).mean())

    # Baseline: shuffle predictions to see correlation by chance
    rng = np.random.RandomState(0)
    shuf = rng.permutation(len(examples))
    chance_r = float(np.corrcoef(r_pred[shuf], r_true)[0, 1])
    chance_g = float(np.corrcoef(g_pred[shuf], g_true)[0, 1])
    chance_b = float(np.corrcoef(b_pred[shuf], b_true)[0, 1])

    print(f"\n  Channel  Pearson r   MAE      shuffled-baseline r")
    print(f"  R        {corr_r:>+.3f}     {mae_r:.3f}    {chance_r:>+.3f}")
    print(f"  G        {corr_g:>+.3f}     {mae_g:.3f}    {chance_g:>+.3f}")
    print(f"  B        {corr_b:>+.3f}     {mae_b:.3f}    {chance_b:>+.3f}")

    # -------------------------------------------------------------------
    # Load supervised baseline from the latest Stage 5 / RGB-probe artifact,
    # so the printed propagation ratio stays in sync with the current model.
    sup_path = Path("data/parsed/stage8/rgb_propagation.json")
    sup_baseline = None
    try:
        # Reuse the comparison-probe direct-read result if present
        cmp = Path("data/parsed/stage8/d5_results.json")
        if cmp.exists():
            import json as _json
            with cmp.open() as _f:
                d = _json.load(_f)
            r = d.get("d5_5_rgb_preserved", {}).get("pearson_r")
            if r:
                sup_baseline = (r["R"], r["G"], r["B"])
    except Exception:
        pass
    if sup_baseline is None:
        # Fall back to the latest published parsed Phase 0 numbers
        sup_baseline = (0.986, 0.986, 0.985)
    sup_R, sup_G, sup_B = sup_baseline

    print(f"\n[4] Comparison to supervised positions (latest parsed Stage 5 baseline):")
    print(f"  Supervised hex/named/rgb positions: R(R,G,B) = "
          f"{sup_R:.3f} / {sup_G:.3f} / {sup_B:.3f}")
    print(f"  Unsupervised HSL positions        : R(R,G,B) = "
          f"{corr_r:.3f} / {corr_g:.3f} / {corr_b:.3f}")
    print(f"  Propagation ratio (HSL / supervised):"
          f" R={corr_r/max(1e-9, sup_R):.2f}  "
          f"G={corr_g/max(1e-9, sup_G):.2f}  "
          f"B={corr_b/max(1e-9, sup_B):.2f}")

    # -------------------------------------------------------------------
    # Save
    results = {
        "n_examples": len(examples),
        "checkpoint": str(args.ckpt),
        "step": ckpt["step"],
        "correlation": {"R": corr_r, "G": corr_g, "B": corr_b},
        "mae": {"R": mae_r, "G": mae_g, "B": mae_b},
        "shuffled_baseline": {"R": chance_r, "G": chance_g, "B": chance_b},
        "supervised_baseline": {"R": sup_R, "G": sup_G, "B": sup_B},
        "supervised_baseline_source": (
            "data/parsed/stage8/d5_results.json (d5_5_rgb_preserved)"
            if sup_baseline != (0.986, 0.986, 0.985)
            else "fallback constant; latest parsed Phase 0 result"
        ),
    }
    out_path = OUT_DIR / "rgb_propagation.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # -------------------------------------------------------------------
    # Verdict
    print(f"\n{'=' * 60}\n  Verdict\n{'=' * 60}")
    avg_corr = (corr_r + corr_g + corr_b) / 3
    if avg_corr >= 0.7:
        print(f"  Strong propagation. Avg |r| = {avg_corr:.3f}")
        print(f"  The named R, G, B dims fire with correct values at HSL positions")
        print(f"  even though the model was never given HSL supervision. The named-")
        print(f"  dimension concept generalizes across CSS color formats.")
    elif avg_corr >= 0.4:
        print(f"  Partial propagation. Avg |r| = {avg_corr:.3f}")
        print(f"  Some signal is propagating; the constraint generalizes weakly.")
    else:
        print(f"  No meaningful propagation. Avg |r| = {avg_corr:.3f}")
        print(f"  The R/G/B dims at HSL positions don't track the HSL's RGB value.")


if __name__ == "__main__":
    main()
