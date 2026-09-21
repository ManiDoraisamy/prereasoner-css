# Pre-registered protocol — locked before training

> **STATUS (2026-09-21): all analyses complete.** 3 anchored seeds (42/43/44)
> + 1 λ=0 baseline trained; A, B, C, D and E all executed as specified.
> Outcomes: **A passed** (hex r = 0.998, rgb-args r = 0.988, both above the
> pre-committed thresholds). **B met the "partial" band** — saturation
> holdout |r| = 0.768, above 0.70 in every seed, so the emergence claim is
> supported *as stated*; but **C and the λ=0 control together show it is not
> attributable to anchoring** (Δ = −0.034 to +0.026, hue better unanchored),
> and index agreement is **0%**. The prediction registered under C — that
> indices would scatter — held. Nothing was revised after seeing results.

Written 2026-09-18, before the vocabulary build, before pre-encoding, before
any of the runs it governs. Every headline number in the paper comes from the
analyses named here, computed the way described here. If a result forces a
deviation, the deviation gets documented as such — the protocol does not get
quietly rewritten.

## The claims under test

> **Corpus note (measured, not estimated).** An early count of 914 hsl files
> was wrong: the pattern `hsl\(` misses `hsla(` and was case-sensitive. The
> true hsl-family pool is **3,686 / 100,000 files (3.7%)**, of which 2,458 go
> to the train shard and 1,228 are held out for probing. Both `hsl()` and
> `hsla()` are parsed by `detect_colors` and usable as probe sites, and
> neither is ever anchored.

1. **Supervised naming works and is compositional.** Dims 0/1/2, anchored to
   R/G/B at colour literals, read the channels of *held-out-file* colours
   directly — hex digit pairs, `rgb()` arguments, named colours.
2. **Unsupervised neighbours emerge.** `hsl()` is never anchored
   (`tests/test_detection.py::NO_HSL_ANCHORS` guards this), yet some
   dimensions come to encode H, S, L at their component tokens.
3. **Emergent indices are not addresses.** Which dims take the H/S/L jobs is
   expected to vary across training seeds even though *that* a few dims do,
   at what strength, is expected to be stable.

## Deviations from the original Phase 0 (all deliberate, all pre-declared)

| # | Change | Why |
|---|---|---|
| D1 | `rgb()`/`rgba()` anchored **per argument** (dim 0 at the R-arg token, 1 at G, 2 at B) instead of one all-three anchor at `)` | Makes "the first three dimensions represent the arguments of rgb()" literally true; same structural scheme as hex pairs. Nested-function calls fall back to the old close-paren anchor. |
| D2 | Force-include integers extended 0-255 → **0-360**; `%` pinned | A hue like `312` must be one atomic token, or the anchor/probe lands on a BPE fragment whose state hasn't read the full number. |
| D3 | **Stratified train shard** (14k files): all training-side hsl-family files + a 6k rgb() quota + random fill | The hsl family (`hsl(`/`hsla(`, case-insensitive) is in 3,686 of 100,000 files — 3.7%. Uniform sampling at a fixed ~20M-token training budget shows the model only a small slice of those; a smaller, denser shard raises hsl exposure **without** changing step count. Shard is ~17.6% hsl files vs 3.7% in the corpus. Composition recorded in `split_meta.json`. Real files only — no synthesis. |
| D4 | **File-level holdout**: probes run only on files excluded from the train shard (⅓ of hsl files, 1500 rgb files, 5000 plain) | Original probes drew from training files. |
| D5 | **Split-half probe protocol**: emergent dims selected on half A of the holdout examples, correlation reported on half B | The original reported the in-sample argmax over 381 dims — selection bias. Both numbers are recorded (`select_r`, `holdout_r`); the headline is `holdout_r`. |
| D6 | Probe/anchor positions use the **last sub-id** of a token | First-fragment probing measured states that hadn't read the value. `--pos-mode first` retained to reproduce archived numbers. |
| D7 | `grad` hue-unit parse fixed ("200grad" matched the `rad` suffix first and was dropped) | Parser bug, caught by the new tests. |
| D8 | DataLoader shuffle driven by an explicit per-seed generator | Shuffle previously derived from global RNG state — reproducible but fragile to code reordering. |
| D9 | Anchor records store the full RGB triple; `channel` alone selects supervision | Placeholder zeros made `anchors.bin` lie about colours (training-equivalent under the mask, but wrong for analysis). |
| D10 | Anchors on BPE-split tokens are dropped and counted, never mis-anchored | Should be ~0 with D2; the count in `meta.json` proves it. |

Model, loss, optimizer, schedule, step count, batch size, context length are
**unchanged** from Phase 0: GPT-2 384/6/6, ctx 1024, MSE anchor (λ=1.0) +
L2 sparsity (λ=0.1) + LM loss, AdamW lr 3e-4, 5000 steps, batch 4.

## The runs

| Run | Seeds | λ_anchor / λ_sparse | Purpose |
|---|---|---|---|
| anchored ×3 | 42, 43, 44 | 1.0 / 0.1 | main result + seed robustness |
| baseline ×1 | 42 | 0.0 / 0.0 | LM-only control, same tokenizer/corpus/steps |

All on the same pre-encoded train shard, same vocabulary (built from the
train shard only — holdout files never influence token frequencies).

## Pre-declared analyses and success criteria

Computed per seed on **holdout files only**; emergent-dim numbers via the
split-half protocol (D5). Chance floors from the shuffled-target control.

**A. Supervised readout** (`stage5`-style + per-arg probe):
- Hex direct-read, held-out files: **pass if r ≥ 0.99** per channel, every seed.
- `rgb()` per-argument direct-read at arg tokens: **pass if r ≥ 0.95**, every
  seed (datatyped8 precedent: 0.985–0.990).
- Named colours: report; expect r ≥ 0.90.
- Baseline (λ=0) linear probe over all 384 dims: expect far below anchored
  direct-read; report the gap.

**B. Emergence** (`seed_robustness.py`, holdout-r):
- Saturation at the S-component token: **claim supported if holdout |r| ≥ 0.70
  in all 3 seeds**; 0.50–0.70 = "partial"; below = claim not supported. Report
  whatever happens.
- Hue (sin/cos basis) and lightness at their component tokens: report; no
  threshold pre-committed (Phase 0 suggests ~0.7 but under a biased protocol).
- Count of dims with select-half |r| ≥ 0.5: report mean ± std across seeds.

**C. Index instability** (the paper's thesis):
- Cross-seed best-dim pairwise agreement: **prediction: ≈ 0** (weak/count
  reading). If indices DO recur across seeds, that is a more interesting
  positive finding and gets reported as such.
- Top-k Jaccard overlap vs simulated chance for k ∈ {5, 10, 25}: report.
- Within-checkpoint bootstrap (500 resamples): modal-dim win rate at n≈1500.
  Prior evidence: stable at n=1500 (95–100%), unstable at n=250.

**D. Controls:**
- Shuffled-target permutation floor (200 perms) reported next to every
  emergent correlation.
- RGB→HSL propagation (dims 0/1/2 read at hsl positions vs the colour's true
  RGB): **prediction: absent** (|r| ≲ 0.3, shuffled baseline 0.02), same as
  Phase 0. The loss still never links the formats.
- Off-anchor leakage: mean |h[0:3]| at non-colour value tokens on holdout
  files; report (no threshold — Phase 0 never measured it).
- LM quality: final LM loss, anchored vs baseline, same corpus — the price
  of the constraint in perplexity terms.

**E. Confirmatory re-run of the original checkpoint** — already done
(`results/REPRODUCTION.md`): archived numbers reproduce to 1e-14 with
`--pos-mode first`.

## What would falsify what

- A(hex or rgb) fails in any seed → the supervised-naming claim as stated is
  wrong for this setup; publish the failure.
- B fails everywhere → "crystallization" was an artifact of in-sample
  selection at Phase 0's sample size; the paper's §3.3 becomes that finding.
- C shows stable indices across seeds → the weak/count thesis is wrong; the
  strong/index claim gains support; rewrite §3.4 accordingly.

Nothing in this protocol depends on which way B or C lands.
