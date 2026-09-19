"""
Stage 8c: Multi-position probe.

Stage 8 measured hidden states at the closing ')' of every hsl(...) call.
The negative result there might be a probe-position artifact: maybe H is
encoded at the hue-component-token position, S at the saturation-component
position, L at the lightness-component position — not aggregated at ')'.

This script probes at FOUR positions per HSL:
  1. The 'hsl(' function-name token (causal position before any args seen)
  2. The H component token (right after 'hsl(')
  3. The S component token
  4. The L component token
  5. The closing ')'  (control — should match Stage 8)

For each probe, it computes per-dim correlations with H, S, L and prints
the best |r| per property. If the dim-localization story is actually true
but at a different probe position, we'll see it here.
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


def find_hsl_components(stream, hsl_pos: int) -> tuple[int, int, int, int] | None:
    """Given a stream and the position of the closing ')' of an hsl/hsla,
    return (func_pos, h_pos, s_pos, l_pos) stream indices.

    The new walker emits `hsl` and `(` as two separate tokens, with split
    components like `240deg` -> `240` + `deg` and `50%` -> `50` + `%`.
    So we:
      1. Walk backward from ')' through balanced parens to find the matching
         '(' and confirm the preceding token is a `hsl` / `hsla` func token.
      2. Collect the FIRST NUMBER token in each comma-separated arg group
         (this is the numeric value; any following unit like `deg` / `%` is
         a separate token and we skip it).
    """
    depth = 1
    open_paren_pos = None
    for i in range(hsl_pos - 1, -1, -1):
        text = stream[i][0]
        if text == ")":
            depth += 1
        elif text == "(":
            depth -= 1
            if depth == 0:
                open_paren_pos = i
                break
    if open_paren_pos is None or open_paren_pos == 0:
        return None
    func_pos = open_paren_pos - 1
    if stream[func_pos][1] != "func" or stream[func_pos][0] not in ("hsl", "hsla"):
        return None

    components: list[int] = []
    expecting_first = True
    for i in range(open_paren_pos + 1, hsl_pos):
        text = stream[i][0]
        ttype = stream[i][1]
        if text == ",":
            expecting_first = True
            continue
        if text == "/":
            break
        if expecting_first and ttype == "number":
            components.append(i)
            expecting_first = False
            if len(components) == 3:
                break
    if len(components) < 3:
        return None
    return (func_pos, components[0], components[1], components[2])


def _map_pos(stream_to_id, ids, stream_pos: int, pos_mode: str,
             add_special: bool = True) -> int:
    """Map a stream position to an ID position.

    pos_mode="last" (correct): the LAST sub-id of the stream token — the
    position at which the model has read the complete token. For atomic
    (fat-head) tokens this equals the first sub-id, so with the 0-360
    integer vocab the two modes coincide on colour tokens.

    pos_mode="first" (legacy): the first sub-id, which is what the original
    Phase 0 analysis used. On a BPE-split hue like "312" this probes a state
    that has only seen "31" — kept solely to reproduce archived results.
    """
    start = stream_to_id[stream_pos]
    if pos_mode == "first":
        return start
    end = (stream_to_id[stream_pos + 1]
           if stream_pos + 1 < len(stream_to_id)
           else len(ids) - (1 if add_special else 0))
    return end - 1


def extract_multiprobe_examples(tok, n_files, max_examples, context_len,
                                css_dir=Path("data/stage1/css"),
                                pos_mode="last", file_list=None):
    """Collect hsl() examples with their component-token positions.

    file_list (a text file of .css paths, one per line) takes precedence
    over css_dir. That is how probes are restricted to HELD-OUT files —
    files excluded from the training shard — per docs/PROTOCOL.md D4.
    """
    if file_list is not None:
        files = [Path(line.strip()) for line in
                 Path(file_list).read_text(encoding="utf-8").splitlines()
                 if line.strip()][:n_files]
    else:
        files = sorted(Path(css_dir).glob("*.css"))[:n_files]
    examples = []
    for path in tqdm(files, desc="Extracting"):
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
            comp = find_hsl_components(stream, h_det.position)
            if comp is None:
                continue
            func_sp, h_sp, s_sp, l_sp = comp
            close_sp = h_det.position
            # Get ID-space positions (see _map_pos for first-vs-last semantics)
            func_id = _map_pos(stream_to_id, ids, func_sp, pos_mode)
            h_id = _map_pos(stream_to_id, ids, h_sp, pos_mode)
            s_id = _map_pos(stream_to_id, ids, s_sp, pos_mode)
            l_id = _map_pos(stream_to_id, ids, l_sp, pos_mode)
            close_id = _map_pos(stream_to_id, ids, close_sp, pos_mode)
            max_id = close_id
            if max_id >= context_len:
                start = max_id - context_len + 1
                window_ids = ids[start: max_id + 1]
                func_id -= start
                h_id -= start
                s_id -= start
                l_id -= start
                close_id -= start
            else:
                end = min(len(ids), context_len)
                window_ids = ids[:end]
            # Sanity: all positions must be inside the window
            if min(func_id, h_id, s_id, l_id, close_id) < 0:
                continue
            if max(func_id, h_id, s_id, l_id, close_id) >= len(window_ids):
                continue
            r, g, b = colorsys.hls_to_rgb(h_det.h / 360.0, h_det.l / 100.0, h_det.s / 100.0)
            examples.append({
                "ids": window_ids,
                "pos_func": func_id, "pos_h": h_id, "pos_s": s_id,
                "pos_l": l_id, "pos_close": close_id,
                "h": float(h_det.h), "s": float(h_det.s), "l": float(h_det.l),
                "r_true": float(r), "g_true": float(g), "b_true": float(b),
            })
            if len(examples) >= max_examples:
                return examples
    return examples


def safe_corr(a, b):
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def best_dim_for_property(activations, target):
    n_dims = activations.shape[1]
    best_r = 0.0
    best_dim = -1
    for d in range(n_dims):
        if d in (0, 1, 2):
            continue
        r = abs(safe_corr(activations[:, d], target))
        if r > best_r:
            best_r = r; best_dim = d
    return best_dim, best_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-files", type=int, default=20000)
    parser.add_argument("--max-examples", type=int, default=1500)
    parser.add_argument("--file-list", default=None,
                        help="Text file of .css paths (overrides the corpus "
                             "glob). Use data/splits/holdout_files.txt.")
    parser.add_argument("--pos-mode", choices=("last", "first"), default="last",
                        help="'last' probes the final sub-id of a component "
                             "token (correct). 'first' reproduces the archived "
                             "Phase 0 numbers (results/multiprobe.json).")
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

    print(f"\n[1] Extracting HSL examples with component positions "
          f"(pos_mode={args.pos_mode})...")
    examples = extract_multiprobe_examples(
        tok, args.n_files, args.max_examples, cfg.context_len,
        pos_mode=args.pos_mode, file_list=args.file_list,
    )
    print(f"  Collected: {len(examples)}")
    if len(examples) < 100:
        print("Not enough examples.")
        return

    H = cfg.hidden_dim
    probes = ("pos_func", "pos_h", "pos_s", "pos_l", "pos_close")
    probe_labels = {
        "pos_func": "'hsl('",
        "pos_h": "H component",
        "pos_s": "S component",
        "pos_l": "L component",
        "pos_close": "')'  (Stage 8 baseline)",
    }
    # Storage: activations[probe_name] shape (n, H)
    acts = {p: np.zeros((len(examples), H), dtype=np.float32) for p in probes}

    print(f"\n[2] Inference, capturing all 5 positions per example...")
    for i, ex in enumerate(tqdm(examples)):
        ids = torch.tensor([ex["ids"]], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model(input_ids=ids)
        hidden = out.hidden_states[-1][0].cpu().numpy()
        for p in probes:
            acts[p][i] = hidden[ex[p]]

    h_vals = np.array([e["h"] for e in examples])
    s_vals = np.array([e["s"] for e in examples])
    l_vals = np.array([e["l"] for e in examples])
    r_true = np.array([e["r_true"] for e in examples])
    g_true = np.array([e["g_true"] for e in examples])
    b_true = np.array([e["b_true"] for e in examples])
    h_sin = np.sin(h_vals * np.pi / 180)
    h_cos = np.cos(h_vals * np.pi / 180)

    print(f"\n[3] Best single-dim |r| per (probe, property)")
    print(f"\n  Test A: do non-RGB dims encode H, S, L locally?")
    print(f"  {'probe':<28}{'H_best':>14}{'S_best':>14}{'L_best':>14}")
    rows = []
    for p in probes:
        # Hue: try sin and cos basis, keep larger
        dh_sin, rh_sin = best_dim_for_property(acts[p], h_sin)
        dh_cos, rh_cos = best_dim_for_property(acts[p], h_cos)
        if rh_sin >= rh_cos:
            dh, rh = dh_sin, rh_sin
        else:
            dh, rh = dh_cos, rh_cos
        ds, rs = best_dim_for_property(acts[p], s_vals)
        dl, rl = best_dim_for_property(acts[p], l_vals)
        print(f"  {probe_labels[p]:<28}"
              f"  {f'd{dh}/{rh:.3f}':>12}"
              f"  {f'd{ds}/{rs:.3f}':>12}"
              f"  {f'd{dl}/{rl:.3f}':>12}")
        rows.append({
            "probe": p, "label": probe_labels[p],
            "best_H_dim": int(dh), "best_H_r": rh,
            "best_S_dim": int(ds), "best_S_r": rs,
            "best_L_dim": int(dl), "best_L_r": rl,
        })

    print(f"\n  Test B: do supervised R/G/B dims (0, 1, 2) encode true RGB at each probe?")
    print(f"  {'probe':<28}{'R':>8}{'G':>8}{'B':>8}")
    rgb_rows = []
    for p in probes:
        rR = safe_corr(acts[p][:, 0], r_true)
        rG = safe_corr(acts[p][:, 1], g_true)
        rB = safe_corr(acts[p][:, 2], b_true)
        print(f"  {probe_labels[p]:<28}  {rR:>+.3f}  {rG:>+.3f}  {rB:>+.3f}")
        rgb_rows.append({"probe": p, "label": probe_labels[p],
                         "r_R": rR, "r_G": rG, "r_B": rB})

    out = {
        "n_examples": len(examples),
        "step": ckpt["step"],
        "best_dim_per_property": rows,
        "rgb_propagation_per_probe": rgb_rows,
    }
    out_path = OUT_DIR / "multiprobe.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 60}\n  Verdict\n{'=' * 60}")
    # Headline: does any probe show |r| > 0.5 for any property?
    found_local = False
    for r in rows:
        for k, v in (("H", r["best_H_r"]), ("S", r["best_S_r"]), ("L", r["best_L_r"])):
            if v >= 0.5:
                print(f"  Localized: {r['label']} encodes {k} at dim {r['best_'+k+'_dim']}, |r|={v:.3f}")
                found_local = True
    if not found_local:
        print("  No probe position localizes H, S, or L to a single non-RGB dim (|r| >= 0.5).")
        print("  The 'distributed encoding' finding from Stage 8 is robust to probe position.")

    found_propagation = False
    for r in rgb_rows:
        avg = (abs(r["r_R"]) + abs(r["r_G"]) + abs(r["r_B"])) / 3
        if avg >= 0.5:
            print(f"  RGB propagation: at {r['label']}, avg |r| with true RGB = {avg:.3f}")
            found_propagation = True
    if not found_propagation:
        print("  No probe position shows R/G/B dims firing with true RGB values.")
        print("  The 'no propagation' finding from Stage 8b is robust to probe position.")


if __name__ == "__main__":
    main()
