# prereasoner-css

**Can you tell a language model what its dimensions are allowed to mean — and
does anything follow from that?**

This repo trains a small transformer on CSS with three of its 384 dimensions
pinned, by an auxiliary loss, to the red, green and blue channels of every
colour literal in the text. Two things come out of it:

1. **The pinned dimensions do what they were told**, and they do it
   *compositionally* — you can read RGB straight off dims 0/1/2 for hex codes
   the model has never seen, at r = 0.996.
2. **Dimensions nobody pinned pick up related jobs on their own.** `hsl()` is
   never anchored — not once, anywhere in the loss — and yet after training,
   dimension 377 tracks saturation, 100 tracks hue, 69 tracks lightness.

(2) is the interesting one, and it is also the one most likely to be
over-read. Most of this repo exists to say precisely how far it goes.

---

## Results

Trained on 100,000 CSS files from `bigcode/the-stack`. GPT-2 architecture,
384 hidden dims, 6 layers, 6 heads, 70,004-token CSS-aware vocabulary,
~38M parameters, 5000 steps.

**Supervised dims — read directly off the hidden state, no probe fitted:**

| | R | G | B |
|---|---|---|---|
| Held-out OOD hex, direct read of dims 0/1/2 | **0.996** | **0.996** | **0.996** |
| Same model, linear probe over all 384 dims | 0.999 | 0.999 | 0.998 |
| Baseline: identical model, LM loss only, probe over 384 dims | 0.551 | 0.285 | 0.654 |

The middle row matters: fitting a probe over all 384 dimensions buys 0.003
over just *reading dimension 0*. The constraint didn't merely make RGB
decodable, it made the other 381 dims irrelevant to RGB. The bottom row is the
control — same architecture, same compute, same corpus, no anchor loss. It
reaches r ≈ 0.50 with a fitted probe, and its G-channel test R² is **−0.739**,
worse than predicting the mean.

**Unsupervised dims — nothing in the loss ever mentions `hsl()`:**

| Property | Probe position | Dim | \|r\| held out |
|---|---|---|---|
| Saturation | S component token | **377** | **0.842** |
| Hue | H component token | **100** | 0.775 |
| Lightness | L component token | **69** | 0.774 |

### What this does *not* show

Stated plainly, because these are easy to lose in the retelling:

- **The three dims are not one readable triple.** Each is the best dim at a
  *different* token position. At one shared position (the closing `)` of
  `hsl(...)`) the numbers fall to S 0.635, H 0.535, L 0.485.
- **Only saturation survives a multivariate held-out test.** R² at the closing
  paren: `S = 0.400`, `cos_H = 0.222`, `L = 0.022`, `sin_H = −0.321`. Hue is
  negative. The honest sentence is "saturation crystallizes; hue and lightness
  show partial structure."
- **RGB does not propagate across formats.** Read dims 0/1/2 at an `hsl()`
  position and compare against that colour's true RGB: |r| = 0.06–0.36, against
  a shuffled baseline of 0.02. The model never learned the conversion. It had no
  reason to — the token after `hsl(240, 50%, 50%)` is almost always `;` or `}`,
  and predicting `;` does not require knowing the colour.
- **Picking the best of 381 dimensions is a multiple-comparisons problem.**
  `probes/seed_robustness.py` includes a shuffled-target control that measures
  the chance-level peak |r| for this n and this dim count. Compare against that
  floor, not against zero.

### The seed question

The numbers above come from **one training run** (seed 42). That leaves two
readings alive, and they are not the same claim:

- **Strong/index** — dimension 100 encodes hue; the index means something.
- **Weak/count** — *some small set* of dimensions comes to encode hue at
  |r| ≈ 0.8, and which indices get the job is arbitrary, a symmetry broken by
  initialization and data order.

We think the weak reading is correct and the strong one is almost certainly
false, but neither is *tested* by a single run. `probes/seed_robustness.py`
trains nothing and settles it: it probes N same-architecture, different-seed
checkpoints on byte-identical HSL examples and reports index agreement, top-k
set overlap against chance, and the spread of the correlation magnitudes.

> **Status: settled.** The weak reading is correct.
>
> | Property | seed 42 | seed 43 | seed 44 | agreement | \|r\| |
> |---|---|---|---|---|---|
> | Hue | 296 | 70 | 29 | **0%** | 0.754 ± 0.026 |
> | Saturation | 93 | 297 | 356 | **0%** | 0.768 ± 0.024 |
> | Lightness | 26 | 263 | 82 | **0%** | 0.794 ± 0.018 |
>
> And the structure is **not caused by the anchoring**: an identical λ=0 model
> scores 0.788 / 0.742 / 0.781 — better than the anchored models for hue.
> A probe over all 381 unanchored dims recovers hue at **94.7%**, against
> 99.6% for anchored red read off one dimension with no probe.
>
> Anchoring does not add the information. It relocates it to an address you
> can read without already knowing the answer.
> See `results/hsl_final_3seed.json` and `results/capacity_compare.json`.

---

## Reproducing

Three tiers, cheapest first.

### 1. Re-derive the published numbers from the checkpoint (minutes)

```bash
pip install -r requirements.txt

python probes/stage8c_multiprobe.py \
    --ckpt data/parsed/stage4/checkpoints/step_005000.pt \
    --n-files 20000 --max-examples 1500
```

Needs the trained checkpoint and the CSS corpus — see
[docs/REPRODUCE.md](docs/REPRODUCE.md) for where to get both.

### 2. Retrain from scratch (~10 h CPU, ~1 h on an A100)

```bash
python pipeline/stage1_download.py --target 100000   # needs HF_TOKEN in .env
python pipeline/stage2_extract_tokens.py
python pipeline/stage2_build_vocab.py
python pipeline/stage4_preencode.py
python pipeline/stage4_train.py --steps 5000 --batch-size 4
```

### 3. The multi-seed study (~10 h per seed)

```bash
python pipeline/run_seeds.py --seeds 42 43 44 --steps 5000 --batch-size 4
python probes/seed_robustness.py \
    --ckpt data/seeds/seed42/step_005000.pt \
    --ckpt data/seeds/seed43/step_005000.pt \
    --ckpt data/seeds/seed44/step_005000.pt
```

Add `--smoke --steps 200` to `run_seeds.py` to check the wiring before
committing days of compute.

---

## Layout

```
src/         model, loss, tokenizer, CSS walker, colour-anchor detection
pipeline/    stage1 download -> stage2 vocab -> stage4 preencode -> train
probes/      stage8* analyses (verbatim) + seed_robustness.py (new)
results/     small result JSONs, checked in
docs/        ERRATA.md, REPRODUCE.md, PAPER.md
```

**Where the constraint lives** — [src/loss.py](src/loss.py):

```python
total = lm_loss + weights.lambda_anchor * anchor_loss + weights.lambda_sparse * sparse_loss
```

`anchor_loss` is MSE between dims 0/1/2 and the true RGB at anchored positions.
`sparse_loss` is L2 on the other 381 dims at those same positions.

**Where `hsl()` is excluded** — [src/color_detection.py](src/color_detection.py),
in `detect_colors`. HSL values are parsed, then routed to a separate `hsls`
list that is returned for *analysis* and never enters `anchors`:

```python
elif name in ("hsl", "hsla") and len(args) >= 3:
    hsls.append(HslDetection(close, name, h, s, l))   # analysis only
```

That line is why the HSL result counts as emergent. If you change one thing
in this repo, check that it still holds.

---

## Provenance

The pipeline is ported near-verbatim from
[`prereasoner-flat-css`](https://github.com/manid/prereasoner-flat-css) Phase 0
(`parsed/`), so the numbers here come from the same code that produced them
there. Changes made during the port:

- Files split into `src/` `pipeline/` `probes/`; each entry point got a 4-line
  `sys.path` shim so the original flat imports still resolve.
- `stage4_train.py` gained `--out-dir` and `--stage2-dir` so seeds don't
  overwrite each other.
- `extract_multiprobe_examples` gained an optional `css_dir` argument,
  defaulting to the original hard-coded path.
- `probes/seed_robustness.py` is new.

No change to the model, the loss, the tokenizer, or the anchor detection.

## Relation to the essay

This repo is the evidence behind one paragraph of *Building intelligence
instead of growing it*. That paragraph, as published, misstates the model's
width and the three HSL dimension indices.
**[docs/ERRATA.md](docs/ERRATA.md)** documents each error with its evidence and
supplies corrected text.
