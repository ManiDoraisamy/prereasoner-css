"""
Analysis A (docs/PROTOCOL.md): do the supervised dims read RGB directly?

Measures, per anchor source, the correlation between the raw hidden
activation at the anchored dim and that channel's true value:

    hex_r/g/b     dim 0/1/2 at the R/G/B digit-pair tokens
    rgb_r/g/b     dim 0/1/2 at the R/G/B ARGUMENT tokens   (new in D1)
    rgba_r/g/b    same, for rgba()
    named         dims 0/1/2 at a named-colour identifier

No probe is fitted. This is a direct read of one number out of the hidden
state — the whole point of naming a dimension. A fitted linear probe would
answer a different (easier) question.

Runs on HELD-OUT files by default (PROTOCOL.md D4): files excluded from the
training shard, so nothing here is measured on data the model trained on.

    python probes/rgb_readout.py \
        --ckpt data/seeds/seed42/step_005000.pt \
        --file-list data/splits/holdout_files.txt

Add --synthetic to additionally test compositional generalization on
uniformly random hex codes, which are almost certainly absent from training
as specific triples.
"""

from __future__ import annotations

# --- repo path shim ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from color_detection import detect_colors
from css_walker import walk_css
from model import NamedDimConfig, NamedDimModel
from tokenizer import CSSTokenizer

CHANNEL_NAMES = ("R", "G", "B")


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = NamedDimConfig(**{k: ckpt["config"][k] for k in
                            ("vocab_size", "hidden_dim", "n_layers",
                             "n_heads", "context_len")})
    model = NamedDimModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, int(ckpt["step"])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def collect_from_files(model, cfg, tok, files, device, max_anchors):
    """One forward pass per file; read every in-window anchor from it."""
    # records[source][channel] -> (measured, target)
    records = defaultdict(lambda: defaultdict(lambda: ([], [])))
    n_anchors = 0

    for path in tqdm(files, desc="Reading", unit="file", leave=False):
        if n_anchors >= max_anchors:
            break
        try:
            css = path.read_text(encoding="utf-8", errors="replace")
            stream = walk_css(css)
            ids, stream_to_id = tok.encode_tokens_with_positions(
                stream, add_special=True)
            anchors, _hsls = detect_colors(stream)
        except Exception:
            continue
        if not anchors:
            continue

        window = ids[: cfg.context_len]
        ids_t = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=ids_t)
        hidden = out.hidden_states[-1][0].cpu().numpy()

        for a in anchors:
            pos = stream_to_id[a.position]
            if pos >= len(window):
                continue
            target_rgb = (a.r, a.g, a.b)
            chans = (0, 1, 2) if a.channel is None else (a.channel,)
            for ch in chans:
                meas, tgt = records[a.source][ch]
                meas.append(float(hidden[pos, ch]))
                tgt.append(float(target_rgb[ch]))
                n_anchors += 1
    return records, n_anchors


def synthetic_hex(model, cfg, tok, device, n, rng):
    """Compositional generalization: uniformly random hex codes.

    Any specific triple is vanishingly unlikely to have been trained on,
    so a high correlation here means the model learned
    (digit-pair position) -> (channel value) as a function, not a lookup.
    """
    meas = {0: [], 1: [], 2: []}
    tgt = {0: [], 1: [], 2: []}
    for _ in tqdm(range(n), desc="Synthetic hex", unit="color", leave=False):
        r, g, b = (int(x) for x in rng.integers(0, 256, 3))
        css = f"a {{ color: #{r:02x}{g:02x}{b:02x}; }}"
        stream = walk_css(css)
        ids, stream_to_id = tok.encode_tokens_with_positions(
            stream, add_special=True)
        anchors, _ = detect_colors(stream)
        if len(anchors) != 3:
            continue
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=ids_t)
        hidden = out.hidden_states[-1][0].cpu().numpy()
        for a in anchors:
            ch = a.channel
            pos = stream_to_id[a.position]
            meas[ch].append(float(hidden[pos, ch]))
            tgt[ch].append((a.r, a.g, a.b)[ch])
    return meas, tgt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--file-list", type=Path,
                   default=Path("data/splits/holdout_files.txt"))
    p.add_argument("--stage2-dir", default="data/parsed/stage2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-files", type=int, default=600)
    p.add_argument("--max-anchors", type=int, default=20000)
    p.add_argument("--synthetic", type=int, default=0,
                   help="Also test N uniformly random hex codes (OOD).")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tok = CSSTokenizer.load(args.stage2_dir)
    model, cfg, step = load_model(args.ckpt, args.device)

    files = [Path(l.strip()) for l in
             args.file_list.read_text(encoding="utf-8").splitlines()
             if l.strip()][: args.n_files]
    print(f"checkpoint step {step}  |  hidden_dim {cfg.hidden_dim}  |  "
          f"{len(files)} held-out files")

    records, n_anchors = collect_from_files(
        model, cfg, tok, files, args.device, args.max_anchors)
    print(f"read {n_anchors:,} anchored channel values\n")

    print(f"{'source':<12}{'ch':>4}{'n':>8}{'Pearson r':>12}{'MAE':>10}")
    print("-" * 46)
    results = {}
    for source in sorted(records):
        for ch in sorted(records[source]):
            meas, tgt = records[source][ch]
            r = pearson(meas, tgt)
            mae = float(np.mean(np.abs(np.array(meas) - np.array(tgt))))
            results[f"{source}/dim{ch}"] = {
                "n": len(meas), "pearson_r": r, "mae": mae,
                "channel": ch, "source": source,
            }
            print(f"{source:<12}{CHANNEL_NAMES[ch]:>4}{len(meas):>8}"
                  f"{r:>12.4f}{mae:>10.4f}")

    synth = None
    if args.synthetic:
        print()
        meas, tgt = synthetic_hex(model, cfg, tok, args.device,
                                  args.synthetic, rng)
        synth = {}
        print(f"{'synthetic OOD hex':<16}{'n':>8}{'Pearson r':>12}{'MAE':>10}")
        print("-" * 46)
        for ch in (0, 1, 2):
            r = pearson(meas[ch], tgt[ch])
            mae = float(np.mean(np.abs(np.array(meas[ch]) - np.array(tgt[ch]))))
            synth[CHANNEL_NAMES[ch]] = {"n": len(meas[ch]),
                                        "pearson_r": r, "mae": mae}
            print(f"{'dim ' + str(ch):<16}{len(meas[ch]):>8}{r:>12.4f}{mae:>10.4f}")

    # PROTOCOL.md analysis A thresholds
    print("\n--- PROTOCOL.md analysis A ---")
    hexr = [results[k]["pearson_r"] for k in results if k.startswith("hex_")]
    rgbr = [results[k]["pearson_r"] for k in results
            if k.startswith(("rgb_", "rgba_"))]
    if hexr:
        print(f"  hex direct-read      min r = {min(hexr):.4f}   "
              f"{'PASS' if min(hexr) >= 0.99 else 'below 0.99 threshold'}")
    if rgbr:
        print(f"  rgb()-arg direct-read min r = {min(rgbr):.4f}   "
              f"{'PASS' if min(rgbr) >= 0.95 else 'below 0.95 threshold'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "checkpoint": str(args.ckpt), "step": step,
            "n_files": len(files), "n_anchor_values": n_anchors,
            "by_source": results, "synthetic_ood_hex": synth,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
