"""
Seed robustness: is the HSL dimension result a property of the method,
or of one lucky training run?

Phase 0 reported that after anchoring dims 0/1/2 to R/G/B, three *unsupervised*
dimensions came to encode the HSL components: dim 100 for hue, dim 377 for
saturation, dim 69 for lightness. Nothing in the loss asked for that --
`hsl()` is never anchored (see color_detection.detect_colors, which routes HSL
into a separate `hsls` list that never enters `anchors`).

That result came from a single training run (seed 42). Two very different
claims are compatible with it:

  (S) STRONG/INDEX: dimension 100 encodes hue. The index means something.
  (W) WEAK/COUNT:   *some small set* of dimensions comes to encode hue, at
                    |r| ~ 0.8. Which indices get the job is arbitrary --
                    a symmetry broken by init and data order.

These make opposite predictions across seeds. (S) predicts the same indices
recur; (W) predicts the indices scatter while the correlation magnitudes and
the number of dims above threshold stay put.

This script trains nothing. It takes N checkpoints that differ only in seed,
probes them all on an IDENTICAL set of HSL examples, and reports:

  1. best dim + |r| per (seed, probe position, property)
  2. index agreement across seeds -- exact-match rate on the best dim
  3. top-k set overlap across seed pairs, vs. the chance rate for k of 381
  4. magnitude stability -- mean +/- std of best |r| across seeds
  5. a SHUFFLED-TARGET CONTROL

(5) matters and is easy to skip. Picking the best of 381 dims is a multiple
comparisons problem: even against pure noise the winner has |r| well above 0.
The control permutes the target values and re-runs the same argmax, which
gives the chance-level best |r| for this n and this dim count. Report any
real number against that floor, not against zero.

Usage
-----
    python probes/seed_robustness.py \
        --ckpt data/seeds/seed42/step_005000.pt \
        --ckpt data/seeds/seed43/step_005000.pt \
        --ckpt data/seeds/seed44/step_005000.pt \
        --out results/seed_robustness.json
"""

from __future__ import annotations

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import NamedDimConfig, NamedDimModel
from tokenizer import CSSTokenizer

from stage8c_multiprobe import (
    best_dim_for_property,
    extract_multiprobe_examples,
    safe_corr,
)

PROBES = ("pos_func", "pos_h", "pos_s", "pos_l", "pos_close")
PROBE_LABELS = {
    "pos_func": "'hsl('",
    "pos_h": "H component",
    "pos_s": "S component",
    "pos_l": "L component",
    "pos_close": "')' (Stage 8 baseline)",
}
# The position at which each property is canonically read -- the Phase 0
# headline numbers (dim 100 / 377 / 69) are each measured at the component
# token for that property, not at one shared position.
CANONICAL_PROBE = {"H": "pos_h", "S": "pos_s", "L": "pos_l"}

# Dims 0/1/2 are supervised, so they are excluded from every "emergent dim"
# search. best_dim_for_property already skips them; this is the count it
# effectively searches over, and the denominator for the chance overlap rate.
N_SEARCHABLE_DIMS = 381


# ---------------------------------------------------------------------------
# Per-checkpoint probing
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = NamedDimConfig(**{k: ckpt["config"][k] for k in
                            ("vocab_size", "hidden_dim", "n_layers",
                             "n_heads", "context_len")})
    model = NamedDimModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt


def collect_activations(model, examples, hidden_dim, device):
    """Run inference once per example, capturing all 5 probe positions."""
    acts = {p: np.zeros((len(examples), hidden_dim), dtype=np.float32)
            for p in PROBES}
    for i, ex in enumerate(tqdm(examples, desc="  inference", leave=False)):
        ids = torch.tensor([ex["ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=ids)
        hidden = out.hidden_states[-1][0].cpu().numpy()
        for p in PROBES:
            acts[p][i] = hidden[ex[p]]
    return acts


def all_dim_correlations(activations, target):
    """|r| against target for every dim. Dims 0/1/2 forced to 0 (supervised).

    Kept as an explicit loop over safe_corr so it is obviously the same
    computation Phase 0 ran. all_dim_correlations_fast is the vectorized
    equivalent used on the hot bootstrap path; check_corr_paths_agree asserts
    the two match before any bootstrap result is reported.
    """
    n_dims = activations.shape[1]
    out = np.zeros(n_dims, dtype=np.float64)
    for d in range(n_dims):
        if d in (0, 1, 2):
            continue
        out[d] = abs(safe_corr(activations[:, d], target))
    return out


def all_dim_correlations_fast(activations, target):
    """Vectorized |r| per dim. Same result as all_dim_correlations, ~100x faster.

    The bootstrap needs tens of thousands of these, which is too many for the
    per-dim Python loop.
    """
    a = activations.astype(np.float64)
    a_c = a - a.mean(axis=0)
    t_c = target.astype(np.float64) - target.mean()
    denom = np.sqrt((a_c ** 2).sum(axis=0) * (t_c ** 2).sum())
    # Zero-variance dims (or target) have no defined correlation -> 0,
    # matching safe_corr's guard.
    safe = denom > 1e-12
    out = np.zeros(a.shape[1], dtype=np.float64)
    out[safe] = np.abs((a_c[:, safe] * t_c[:, None]).sum(axis=0) / denom[safe])
    out[:3] = 0.0
    return out


def check_corr_paths_agree(activations, target, tol=1e-6):
    """Fail loudly if the fast path ever diverges from the audited loop."""
    slow = all_dim_correlations(activations, target)
    fast = all_dim_correlations_fast(activations, target)
    delta = float(np.max(np.abs(slow - fast)))
    if delta > tol:
        raise SystemExit(
            f"correlation paths disagree by {delta:.2e} (tol {tol:.0e}); "
            "refusing to report bootstrap numbers"
        )
    return delta


def property_targets(examples):
    h = np.array([e["h"] for e in examples])
    # Hue is circular: 359deg and 1deg are neighbours, so a raw correlation
    # against degrees is meaningless. Phase 0 projects onto sin/cos and keeps
    # whichever basis correlates more strongly; we reproduce that exactly.
    return {
        "H_sin": np.sin(h * np.pi / 180),
        "H_cos": np.cos(h * np.pi / 180),
        "S": np.array([e["s"] for e in examples]),
        "L": np.array([e["l"] for e in examples]),
    }


def probe_checkpoint(model, cfg, examples, device, split):
    """Per-probe, per-property: select the best dim on split A, report on B.

    Selecting the argmax over 381 dims and quoting its correlation on the
    SAME examples is a selection procedure — the quoted number is biased
    upward, and the original Phase 0 analysis did exactly that. Here:

      - `select_r` : |r| of the winning dim on the selection half (A).
        This is the biased, Phase-0-comparable number.
      - `holdout_r`: |r| of that SAME dim on the untouched half (B).
        This is the honest, headline number.

    The hue sin/cos basis choice is also made on A only.
    """
    idx_a, idx_b = split
    acts = collect_activations(model, examples, cfg.hidden_dim, device)
    tgts = property_targets(examples)

    result = {}
    for p in PROBES:
        a = acts[p]
        # Hue: choose the stronger basis on the SELECTION half only.
        r_sin_a = all_dim_correlations(a[idx_a], tgts["H_sin"][idx_a])
        r_cos_a = all_dim_correlations(a[idx_a], tgts["H_cos"][idx_a])
        if r_sin_a.max() >= r_cos_a.max():
            h_vec_a, h_target = r_sin_a, tgts["H_sin"]
        else:
            h_vec_a, h_target = r_cos_a, tgts["H_cos"]

        per_prop = {
            "H": (h_vec_a, h_target),
            "S": (all_dim_correlations(a[idx_a], tgts["S"][idx_a]), tgts["S"]),
            "L": (all_dim_correlations(a[idx_a], tgts["L"][idx_a]), tgts["L"]),
        }

        result[p] = {}
        for prop, (vec_a, target) in per_prop.items():
            best = int(np.argmax(vec_a))
            holdout_r = abs(safe_corr(a[idx_b, best], target[idx_b]))
            result[p][prop] = {
                "best_dim": best,
                "select_r": float(np.max(vec_a)),
                "holdout_r": float(holdout_r),
                "n_above_0.5": int((vec_a >= 0.5).sum()),
                "n_above_0.7": int((vec_a >= 0.7).sum()),
                "all_r": vec_a,        # selection-half vector, for overlap stats
            }
    # acts is returned so the bootstrap and the shuffled control can reuse it
    # instead of paying for another full inference pass.
    return result, acts


def bootstrap_index_stability(acts, examples, n_boot, rng):
    """Does the winning dim survive resampling the examples, model held fixed?

    This isolates a different source of instability from the seed study. There
    the model changes; here it does not. If the argmax over 381 dims moves
    just because you drew a different sample of HSL literals, then "dimension
    100 is hue" is partly a statement about the evaluation set, and no amount
    of seed-averaging will rescue the index.

    Reports, per property, the modal winning dim and how often it wins.
    """
    tgts = property_targets(examples)
    n = len(examples)
    out = {}

    for prop, probe in CANONICAL_PROBE.items():
        a = acts[probe]
        if prop == "H":
            # Match the headline convention: pick the stronger hue basis once
            # on the full sample, then hold that basis fixed across resamples.
            r_sin = all_dim_correlations_fast(a, tgts["H_sin"])
            r_cos = all_dim_correlations_fast(a, tgts["H_cos"])
            target = tgts["H_sin"] if r_sin.max() >= r_cos.max() else tgts["H_cos"]
        else:
            target = tgts[prop]

        check_corr_paths_agree(a, target)
        full_best = int(np.argmax(all_dim_correlations_fast(a, target)))

        winners = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            winners.append(int(np.argmax(
                all_dim_correlations_fast(a[idx], target[idx])
            )))

        counts = {}
        for w in winners:
            counts[w] = counts.get(w, 0) + 1
        modal_dim = max(counts, key=counts.get)

        out[prop] = {
            "probe": probe,
            "full_sample_best_dim": full_best,
            "n_boot": n_boot,
            "n_distinct_winners": len(counts),
            "modal_dim": modal_dim,
            "modal_win_rate": counts[modal_dim] / n_boot,
            "full_sample_dim_win_rate": counts.get(full_best, 0) / n_boot,
            "top_winners": sorted(
                ({"dim": d, "wins": c} for d, c in counts.items()),
                key=lambda x: -x["wins"],
            )[:8],
        }
    return out


def shuffled_control(acts, examples, n_perm, rng):
    """Chance-level best |r| when the target is permuted against the activations.

    Same argmax over the same 381 dims, so this is the floor that a real
    result has to clear. Uses the S-component position, where the real
    signal is strongest -- the hardest place to beat by luck.
    """
    s_vals = np.array([e["s"] for e in examples])
    check_corr_paths_agree(acts["pos_s"], s_vals)
    peaks = []
    for _ in tqdm(range(n_perm), desc="  shuffled control", leave=False):
        permuted = rng.permutation(s_vals)
        peaks.append(float(np.max(all_dim_correlations_fast(acts["pos_s"], permuted))))
    return peaks


# ---------------------------------------------------------------------------
# Cross-seed statistics
# ---------------------------------------------------------------------------

def chance_jaccard(k, n_dims=N_SEARCHABLE_DIMS, n_sim=20000, seed=0):
    """Expected Jaccard of two independent uniform k-subsets of n_dims.

    Estimated by simulation rather than closed form. The obvious analytic
    approximation, k/(2N-k), substitutes E[X]/E[Y] for E[X/Y] and runs ~12%
    low at k=5 -- which would make a chance-level overlap look better than
    chance. Wrong direction for a baseline, so we simulate.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_sim):
        a = set(rng.choice(n_dims, k, replace=False).tolist())
        b = set(rng.choice(n_dims, k, replace=False).tolist())
        total += len(a & b) / len(a | b)
    return total / n_sim


def topk_overlap(per_seed_vectors, k):
    """Mean pairwise Jaccard of top-k dim sets, against the chance baseline."""
    sets = [set(np.argsort(-v)[:k].tolist()) for v in per_seed_vectors]
    jaccards = [
        len(a & b) / len(a | b)
        for a, b in combinations(sets, 2)
    ]
    return {
        "k": k,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else None,
        "chance_jaccard": float(chance_jaccard(k)),
        "pairwise": [float(j) for j in jaccards],
    }


def summarize(per_seed, seed_labels):
    """Index stability vs magnitude stability, per property.

    Magnitude statistics use HOLDOUT correlations (dim selected on half A,
    r measured on half B). Selection-half values are kept per seed for
    comparison with the original Phase 0 in-sample numbers.
    """
    summary = {}
    for prop, probe in CANONICAL_PROBE.items():
        best_dims = [s[probe][prop]["best_dim"] for s in per_seed]
        select_rs = [s[probe][prop]["select_r"] for s in per_seed]
        holdout_rs = [s[probe][prop]["holdout_r"] for s in per_seed]
        above_05 = [s[probe][prop]["n_above_0.5"] for s in per_seed]
        vectors = [s[probe][prop]["all_r"] for s in per_seed]

        n_pairs = len(list(combinations(range(len(best_dims)), 2)))
        n_agree = sum(1 for a, b in combinations(best_dims, 2) if a == b)

        summary[prop] = {
            "probe": probe,
            "probe_label": PROBE_LABELS[probe],
            "per_seed": [
                {"seed": lbl, "best_dim": d, "select_r": sr,
                 "holdout_r": hr, "n_above_0.5": n}
                for lbl, d, sr, hr, n in zip(
                    seed_labels, best_dims, select_rs, holdout_rs, above_05)
            ],
            "index_stability": {
                "unique_dims": len(set(best_dims)),
                "n_seeds": len(best_dims),
                "pairwise_agreement": (n_agree / n_pairs) if n_pairs else None,
            },
            "magnitude_stability": {
                "mean_holdout_r": float(np.mean(holdout_rs)),
                "std_holdout_r": float(np.std(holdout_rs)),
                "min_holdout_r": float(np.min(holdout_rs)),
                "max_holdout_r": float(np.max(holdout_rs)),
                "mean_select_r": float(np.mean(select_rs)),
            },
            "count_stability": {
                "mean_n_above_0.5": float(np.mean(above_05)),
                "std_n_above_0.5": float(np.std(above_05)),
            },
            "topk_overlap": [topk_overlap(vectors, k) for k in (5, 10, 25)],
        }
    return summary


def strip_vectors(per_seed):
    """Drop the 384-float vectors before writing JSON."""
    return [
        {p: {prop: {k: v for k, v in d.items() if k != "all_r"}
             for prop, d in probes.items()}
         for p, probes in seed.items()}
        for seed in per_seed
    ]


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, action="append", required=True,
                        help="Checkpoint path; repeat once per seed.")
    parser.add_argument("--label", action="append", default=None,
                        help="Label per checkpoint (default: parent dir name).")
    parser.add_argument("--stage2-dir", default="data/parsed/stage2")
    parser.add_argument("--css-dir", default="data/stage1/css")
    parser.add_argument("--file-list", default=None,
                        help="Text file of .css paths (overrides --css-dir). "
                             "Use data/splits/holdout_files.txt so every "
                             "number is computed on files the model never "
                             "trained on (PROTOCOL.md D4).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-files", type=int, default=20000)
    parser.add_argument("--max-examples", type=int, default=1500)
    parser.add_argument("--n-perm", type=int, default=50,
                        help="Permutations for the shuffled-target control.")
    parser.add_argument("--n-boot", type=int, default=500,
                        help="Resamples for the bootstrap index-stability test.")
    parser.add_argument("--out", type=Path, default=Path("results/seed_robustness.json"))
    args = parser.parse_args()

    # One checkpoint is enough for the bootstrap and the shuffled control,
    # which are properties of a single model. The cross-seed statistics need
    # two or more and are skipped otherwise.
    labels = args.label or [c.parent.name for c in args.ckpt]
    if len(labels) != len(args.ckpt):
        parser.error("--label count must match --ckpt count")

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    tok = CSSTokenizer.load(args.stage2_dir)

    # Examples are checkpoint-independent, so extract once. Every seed is then
    # probed on byte-identical inputs and any difference is the model, not the
    # sample.
    print(f"[1] Extracting HSL examples (shared across all seeds)...")
    _, cfg0, _ = load_model(args.ckpt[0], args.device)
    examples = extract_multiprobe_examples(
        tok, args.n_files, args.max_examples, cfg0.context_len,
        css_dir=args.css_dir, file_list=args.file_list,
    )
    print(f"    {len(examples)} examples")
    if len(examples) < 200:
        print("Not enough examples (need >= 200 for split-half); aborting.")
        return

    # One fixed selection/holdout split, shared by every seed: dims are
    # SELECTED on half A and REPORTED on half B. The split seed is a
    # constant, not derived from the model seeds.
    n = len(examples)
    perm = np.random.default_rng(1).permutation(n)
    split = (np.sort(perm[: n // 2]), np.sort(perm[n // 2:]))
    print(f"    split: {len(split[0])} selection / {len(split[1])} holdout")

    print(f"\n[2] Probing {len(args.ckpt)} checkpoints...")
    per_seed, steps, acts0 = [], [], None
    for path, lbl in zip(args.ckpt, labels):
        print(f"  {lbl}: {path}")
        model, cfg, ckpt = load_model(path, args.device)
        if cfg.hidden_dim != cfg0.hidden_dim:
            raise SystemExit(
                f"hidden_dim mismatch: {lbl} has {cfg.hidden_dim}, "
                f"expected {cfg0.hidden_dim}. Seeds must share architecture."
            )
        result, acts = probe_checkpoint(model, cfg, examples, args.device, split)
        per_seed.append(result)
        steps.append(int(ckpt["step"]))
        if acts0 is None:
            acts0 = acts

    print(f"\n[3] Bootstrap index stability on {labels[0]} "
          f"({args.n_boot} resamples, model held fixed)...")
    bootstrap = bootstrap_index_stability(acts0, examples, args.n_boot, rng)

    print(f"[4] Shuffled-target control on {labels[0]}...")
    control_peaks = shuffled_control(acts0, examples, args.n_perm, rng)
    control = {
        "n_perm": args.n_perm,
        "probe": "pos_s",
        "property": "S",
        "mean_peak_r": float(np.mean(control_peaks)),
        "p95_peak_r": float(np.percentile(control_peaks, 95)),
        "max_peak_r": float(np.max(control_peaks)),
    }

    summary = summarize(per_seed, labels) if len(per_seed) >= 2 else None

    # ---------------- report ----------------
    print(f"\n{'=' * 72}\n  Robustness\n{'=' * 72}")
    print(f"\nChance floor (shuffled targets, best of {N_SEARCHABLE_DIMS} dims, "
          f"n={len(examples)}):")
    print(f"  mean peak |r| = {control['mean_peak_r']:.3f}   "
          f"p95 = {control['p95_peak_r']:.3f}   max = {control['max_peak_r']:.3f}")

    print(f"\nBootstrap index stability ({labels[0]}, model fixed, "
          f"{args.n_boot} resamples):")
    print(f"  {'prop':<6}{'full-sample':>13}{'modal':>8}{'win rate':>11}"
          f"{'distinct':>10}")
    for prop, b in bootstrap.items():
        print(f"  {prop:<6}{b['full_sample_best_dim']:>13}"
              f"{b['modal_dim']:>8}{b['modal_win_rate']:>10.0%}"
              f"{b['n_distinct_winners']:>10}")
    print("  (low win rate / many distinct winners => the index is a property\n"
          "   of the sample, not of the model)")

    if summary is None:
        print("\nOnly one checkpoint given; skipping cross-seed statistics.")
        print("Pass --ckpt twice or more to test index stability across seeds.")

    for prop, s in (summary or {}).items():
        print(f"\n--- {prop} at {s['probe_label']} ---")
        for row in s["per_seed"]:
            print(f"  {row['seed']:<12} best dim {row['best_dim']:>4}   "
                  f"select |r| = {row['select_r']:.3f}   "
                  f"holdout |r| = {row['holdout_r']:.3f}   "
                  f"dims>=0.5: {row['n_above_0.5']}")
        idx, mag = s["index_stability"], s["magnitude_stability"]
        print(f"  index   : {idx['unique_dims']} distinct dims across "
              f"{idx['n_seeds']} seeds  "
              f"(pairwise agreement {idx['pairwise_agreement']:.0%})")
        print(f"  holdout magnitude: |r| = {mag['mean_holdout_r']:.3f} "
              f"+/- {mag['std_holdout_r']:.3f}  "
              f"[{mag['min_holdout_r']:.3f}, {mag['max_holdout_r']:.3f}]")
        for ov in s["topk_overlap"]:
            print(f"  top-{ov['k']:<3} overlap: Jaccard "
                  f"{ov['mean_jaccard']:.3f} vs chance {ov['chance_jaccard']:.3f}")

    out = {
        "n_examples": len(examples),
        "checkpoints": [str(c) for c in args.ckpt],
        "labels": labels,
        "steps": steps,
        "hidden_dim": cfg0.hidden_dim,
        "n_searchable_dims": N_SEARCHABLE_DIMS,
        "shuffled_control": control,
        "bootstrap_index_stability": bootstrap,
        "summary": summary,
        "per_seed_raw": strip_vectors(per_seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
