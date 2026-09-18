"""
Stage 8d: Hex-code generalization under the new digit-pair tokenizer.

Under the original tokenizer, hex codes were one fused token; only the
most frequent 4,704 hex codes made the fat head. OOD hex codes (synthesized
random ones not in the fat head) BPE-split, and the model lost track of
RGB completely — r ≈ 0 for OOD vs r ≈ 0.55 for IN-distribution.

Under the new tokenizer:
  - Every hex code expands to '#' + three digit-pair tokens (RR, GG, BB)
  - All 256 digit pairs (00..ff) are force-included in the fat head
  - Per-channel constraint anchors during training: dim 0 supervised at the
    R-pair position, dim 1 at G-pair, dim 2 at B-pair

If the model has learned the structural mapping, ANY hex code (seen or
unseen as a specific triple) should produce:
  - dim 0 activation at the R-pair position ~ correct R value
  - dim 1 activation at the G-pair position ~ correct G value
  - dim 2 activation at the B-pair position ~ correct B value

We test on random hex codes drawn uniformly across the RGB cube. The
in-distribution vs OOD distinction largely disappears under the new
tokenizer since every hex is tokenized the same way.
"""

from __future__ import annotations

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import random
from pathlib import Path

import numpy as np
import torch

from css_walker import walk_css
from model import NamedDimConfig, NamedDimModel
from tokenizer import CSSTokenizer

CKPT_DIR = Path("data/parsed/stage4/checkpoints")


def latest_checkpoint() -> Path:
    ckpts = sorted(CKPT_DIR.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {CKPT_DIR}")
    return ckpts[-1]


def find_hex_pair_positions(stream, hex_text: str) -> tuple[int, int, int] | None:
    """Find the three digit-pair token positions for the first hex code
    matching `hex_text` (with or without `#`). Returns (r_pos, g_pos, b_pos)
    in stream-index space, or None if not found."""
    body = hex_text[1:] if hex_text.startswith("#") else hex_text
    body = body.lower()
    if len(body) == 3:
        body = body[0]*2 + body[1]*2 + body[2]*2
    elif len(body) == 4:
        body = body[0]*2 + body[1]*2 + body[2]*2
    elif len(body) == 8:
        body = body[:6]
    r, g, b = body[0:2], body[2:4], body[4:6]
    # Walk the stream for '#' followed by three hash-typed tokens
    for i in range(len(stream) - 3):
        if (stream[i][0] == "#" and stream[i][1] == "literal"
            and stream[i+1][0] == r and stream[i+1][1] == "hash"
            and stream[i+2][0] == g and stream[i+2][1] == "hash"
            and stream[i+3][0] == b and stream[i+3][1] == "hash"):
            return (i+1, i+2, i+3)
    return None


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    body = h[1:] if h.startswith("#") else h
    body = body.lower()
    if len(body) == 3:
        body = "".join(c * 2 for c in body)
    elif len(body) == 4:
        body = "".join(c * 2 for c in body[:3])
    elif len(body) == 8:
        body = body[:6]
    return (int(body[0:2], 16) / 255.0,
            int(body[2:4], 16) / 255.0,
            int(body[4:6], 16) / 255.0)


def run_one(model, tok, hex_code):
    css = f".x {{ color: {hex_code}; }}"
    stream = walk_css(css)
    ids, stream_to_id = tok.encode_tokens_with_positions(stream, add_special=True)
    pair_positions = find_hex_pair_positions(stream, hex_code)
    if pair_positions is None:
        return None
    r_sp, g_sp, b_sp = pair_positions
    r_id, g_id, b_id = stream_to_id[r_sp], stream_to_id[g_sp], stream_to_id[b_sp]
    inp = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=inp)
    hidden = out.hidden_states[-1][0].numpy()    # (T, H)
    # Channel-specific reading: dim 0 at R-pair, dim 1 at G-pair, dim 2 at B-pair
    return {
        "r_pred": float(hidden[r_id, 0]),
        "g_pred": float(hidden[g_id, 1]),
        "b_pred": float(hidden[b_id, 2]),
        # Full hidden at R-pair for other-dim profiling
        "hidden_r": hidden[r_id],
        "hidden_g": hidden[g_id],
        "hidden_b": hidden[b_id],
    }


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    rng = random.Random(0)

    tok = CSSTokenizer.load("data/parsed/stage2")
    ckpt_path = latest_checkpoint()
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = NamedDimConfig(**{k: ckpt["config"][k] for k in
                            ("vocab_size", "hidden_dim", "n_layers", "n_heads", "context_len")})
    model = NamedDimModel(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  step={ckpt['step']}  hidden={cfg.hidden_dim}")

    # Build a uniform-random hex test set (no IN vs OOD distinction needed
    # under the new tokenizer)
    n_test = 500
    hex_codes: list[str] = []
    while len(hex_codes) < n_test:
        h = "#{:02x}{:02x}{:02x}".format(
            rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255),
        )
        if h not in hex_codes:
            hex_codes.append(h)
    print(f"\nTest set: {len(hex_codes)} random hex codes")

    r_pred, g_pred, b_pred = [], [], []
    r_true, g_true, b_true = [], [], []
    hidden_r_all, hidden_g_all, hidden_b_all = [], [], []
    for h in hex_codes:
        result = run_one(model, tok, h)
        if result is None:
            continue
        r_pred.append(result["r_pred"])
        g_pred.append(result["g_pred"])
        b_pred.append(result["b_pred"])
        tr, tg, tb = hex_to_rgb(h)
        r_true.append(tr); g_true.append(tg); b_true.append(tb)
        hidden_r_all.append(result["hidden_r"])
        hidden_g_all.append(result["hidden_g"])
        hidden_b_all.append(result["hidden_b"])

    r_pred = np.array(r_pred); r_true = np.array(r_true)
    g_pred = np.array(g_pred); g_true = np.array(g_true)
    b_pred = np.array(b_pred); b_true = np.array(b_true)
    hidden_r = np.array(hidden_r_all)
    hidden_g = np.array(hidden_g_all)
    hidden_b = np.array(hidden_b_all)

    print(f"\n=== Per-channel encoding accuracy ===")
    print(f"Probe : dim 0 @ R-pair token, dim 1 @ G-pair token, dim 2 @ B-pair token")
    print()
    for label, pred, true in [("R", r_pred, r_true), ("G", g_pred, g_true), ("B", b_pred, b_true)]:
        rr = float(np.corrcoef(pred, true)[0, 1])
        mae = float(np.abs(pred - true).mean())
        print(f"  {label}: Pearson r = {rr:+.3f}    MAE = {mae:.3f}")

    print(f"\n=== Other-dim activity profile ===")
    print(f"At R-pair token:")
    print(f"  mean |act| dim 0 (R)   : {abs(hidden_r[:, 0]).mean():.3f}")
    print(f"  mean |act| dims 1,2    : {np.abs(hidden_r[:, 1:3]).mean():.3f}")
    print(f"  mean |act| dims 3..end : {np.abs(hidden_r[:, 3:]).mean():.3f}")
    print(f"At G-pair token:")
    print(f"  mean |act| dim 1 (G)   : {abs(hidden_g[:, 1]).mean():.3f}")
    print(f"  mean |act| dims 0,2    : {np.abs(hidden_g[:, [0, 2]]).mean():.3f}")
    print(f"  mean |act| dims 3..end : {np.abs(hidden_g[:, 3:]).mean():.3f}")
    print(f"At B-pair token:")
    print(f"  mean |act| dim 2 (B)   : {abs(hidden_b[:, 2]).mean():.3f}")
    print(f"  mean |act| dims 0,1    : {np.abs(hidden_b[:, :2]).mean():.3f}")
    print(f"  mean |act| dims 3..end : {np.abs(hidden_b[:, 3:]).mean():.3f}")

    # Comparison to old test
    avg_corr = (float(np.corrcoef(r_pred, r_true)[0, 1])
                + float(np.corrcoef(g_pred, g_true)[0, 1])
                + float(np.corrcoef(b_pred, b_true)[0, 1])) / 3
    print(f"\n=== Verdict ===")
    print(f"Avg per-channel Pearson r : {avg_corr:.3f}")
    print(f"  (old tokenizer baseline : 0.55 for in-distribution, -0.03 for OOD)")
    if avg_corr >= 0.7:
        print(f"  -> STRONG generalization: structural digit-pair -> channel mapping learned.")
    elif avg_corr >= 0.4:
        print(f"  -> Moderate generalization.")
    else:
        print(f"  -> Weak/no generalization.")


if __name__ == "__main__":
    main()
