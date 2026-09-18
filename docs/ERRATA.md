# Errata: the CSS paragraph in *Building intelligence instead of growing it*

The essay contains one paragraph describing the CSS experiment. The mechanism
it describes is real and reproduces. Three specific details in it are wrong.
This document states each error, gives the evidence, and supplies corrected
text.

## The paragraph as published

> To understand this better, I trained a 1024 dimensional AI model with CSS.
> During training, I constrained the first three dimensions to represent the
> arguments of CSS's rgb() function. After training, those dimensions
> represented red, green, and blue as intended. More surprisingly, dimensions
> 103, 106, 108 automatically crystallized for hsl() function in CSS.

---

## Error 1 — "1024 dimensional" (the model is 384-dimensional)

`1024` is the **context length**, not the embedding width.

```python
# src/model.py
class NamedDimConfig:
    vocab_size:  int = 70_004
    hidden_dim:  int = 384      # n_embd      <- the model's dimensionality
    n_layers:    int = 6
    n_heads:     int = 6
    context_len: int = 1024     # n_positions <- this is the 1024
```

Corroborated by the result file itself, which records the width it searched:

```json
// data/parsed/stage8/discovery_results.json
"hidden_dim": 384
```

No 1024-wide model exists anywhere in the work. Every phase (0, 0.5, 0.6, 0.7,
0.8) uses `hidden_dim = 384`, 6 layers, 6 heads, ~38M parameters.

## Error 2 — "dimensions 103, 106, 108" (they are 377, 100, 69)

The measured result:

| Property | Probe position | Dim | \|r\| (held out) |
|---|---|---|---|
| Saturation | S component token | **377** | 0.842 |
| Hue | H component token | **100** | 0.775 |
| Lightness | L component token | **69** | 0.774 |

Source: `data/parsed/stage8/multiprobe.json`, produced by
`probes/stage8c_multiprobe.py` against checkpoint `step_005000.pt`.

These were re-derived from scratch on 2026-09-18: every dimension index came
back identical and every correlation matched the original to ~1e-14. See
[../results/REPRODUCTION.md](../results/REPRODUCTION.md).

Dimensions **103 and 106 appear nowhere** in the HSL results. Dim 108 appears
only as one of 88 saturation candidates, not as a best dim.

The likely origin of the mix-up: dims 101/102/103 *are* meaningful in this
project — but in a **different experiment** (Phase 0.5/0.6/0.7), where they are
the **supervised** font-shorthand dims, not emergent colour dims. `PHASE.7.md`:
"anchored atomic property dims (font-weight at dim 101, etc.)". Those are told
what to mean; the HSL dims are not.

## Error 3 — "the arguments of CSS's rgb() function" (mostly hex, not rgb())

The constraint fires at three kinds of site, and `rgb()` is the smallest:

- **hex literals** — per-channel: dim 0 anchored at the R digit-pair token, dim 1
  at the G pair, dim 2 at the B pair. Three anchors per colour. This is the bulk
  of the 27.6M anchors and the source of the headline r = 0.996.
- **named colours** (`red`, `teal`) — one anchor on all three dims.
- **`rgb()` / `rgba()`** — one anchor on all three dims, at the closing `)`.

Anchoring only `rgb()` arguments would supply a small fraction of the gradient
signal and would not produce the compositional hex→RGB mapping, which is the
part that generalizes to unseen colours.

---

## Two omissions worth fixing while you're in there

**The three dims are not one readable triple.** Each is the best dim at a
*different* token position. At the single position originally specified (the
closing `)` of `hsl(...)`), the result is markedly weaker: S 0.635, H 0.535,
L 0.485. "Crystallized for hsl()" oversells a set of position-specific
correlational probes.

**Only saturation is clean.** Multivariate held-out R² at the closing paren:

```json
{"sin_H": -0.321, "cos_H": 0.222, "S": 0.400, "L": 0.022}
```

Hue is *negative* and lightness is ~0. The defensible claim is "saturation
crystallizes; hue and lightness show partial structure."

**RGB does not propagate to HSL.** The R/G/B dims read at `hsl(...)` positions
correlate with the colour's true RGB at |r| = 0.06–0.36, against a shuffled
baseline of 0.02. The model never learned to convert. The source repo is
candid about this; the essay omits it, and a reader who runs the code will
find it.

---

## Corrected paragraph

> To understand this better, I trained a small transformer — 384-dimensional,
> 38 million parameters — on 100,000 CSS files. During training I constrained
> the first three dimensions to carry the red, green and blue channels of every
> colour the model read: dimension 0 anchored at the red digit-pair of each hex
> code, dimension 1 at the green pair, dimension 2 at the blue pair, plus the
> same three dimensions at named colours and `rgb()` calls. After training those
> dimensions represented red, green and blue as intended — reading them directly
> off a hex code the model had never seen recovers its channel values at
> r = 0.996. More surprisingly, nothing in the loss ever mentioned `hsl()`, and
> yet other dimensions took the job on unprompted: dimension 377 came to track
> saturation (|r| = 0.84), dimension 100 hue (0.77), dimension 69 lightness
> (0.77). I should be precise about what that does and doesn't show. Saturation
> is the only one that holds up under a multivariate held-out test; hue and
> lightness show partial structure. And this is one training run — which
> dimensions get the job is almost certainly an accident of initialization,
> even if the fact that some small set of them does is not.

### Shorter version, if the essay needs the space

> To understand this better, I trained a small 384-dimensional transformer on
> 100,000 CSS files, constraining its first three dimensions to carry the red,
> green and blue channels of every colour literal it read. After training those
> dimensions did represent red, green and blue — reading them off an unseen hex
> code recovers its channels at r = 0.996. More surprisingly, nothing in the
> loss ever mentioned `hsl()`, yet other dimensions crystallized for it anyway:
> dimension 377 for saturation, 100 for hue, 69 for lightness.

---

## Reproducing these numbers

The checkpoint that produced them still exists. No retraining needed:

```bash
python probes/stage8c_multiprobe.py \
    --ckpt data/parsed/stage4/checkpoints/step_005000.pt \
    --n-files 20000 --max-examples 1500
```

See [REPRODUCE.md](REPRODUCE.md) for the full path from raw corpus to result.
