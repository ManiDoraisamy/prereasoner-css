# You can name a dimension, but you cannot find one

**Supervised anchoring produces compositional named dimensions in a CSS
language model; the emergent ones have no stable address.**

*Draft. Sections marked ⏳ await the multi-seed run.*

---

## Abstract

Interpretability is usually retrofitted: train a model, then go looking for
what its dimensions mean. We invert this for a narrow domain. We train a 38M
parameter transformer on 100,000 CSS files with an auxiliary loss that pins
three of its 384 hidden dimensions to the red, green and blue channels of
every colour literal in the text.

Three findings. **First**, the supervision works, and it works
*compositionally*: reading dimensions 0/1/2 directly off the hidden state
recovers the channel values of hex codes never seen in training at r = 0.996,
against r = 0.497 for an identical model trained with language-modelling loss
alone and probed with a fitted linear map over all 384 dimensions. Structured
tokenization is doing essential work here — the effect collapses when hex codes
are tokenized as opaque units.

**Second**, dimensions that nobody supervised take on neighbouring jobs.
`hsl()` is never anchored anywhere in the loss, yet after training we find
single dimensions correlating with saturation at |r| = 0.84, hue at 0.78 and
lightness at 0.77, each at its own component-token position.

**Third**, and the point of the paper: *which* dimensions those are is not a
fact about the model. Resampling the evaluation set — with the model held
completely fixed — moves the winning index. So does changing the seed. What is
stable is the *count* and the *magnitude*: a small number of dimensions reach
|r| ≈ 0.8, reliably, at indices that carry no information across runs.

The practical consequence for interpretability-by-construction: supervision
buys you an address, and emergence does not. If you want a dimension you can
name, you have to name it.

---

## 1. Introduction

Dario Amodei's framing is that we grow models rather than build them, which is
why interpretability is hard and urgent. The retrofitted approach — train
first, interpret later — has an inherent difficulty: nothing in training made
any particular direction in activation space correspond to any particular human
concept, so the interpreter's job is to discover a code that was never designed
to be read.

An alternative is to fix part of the code in advance. Reserve dimensions,
declare what they mean, and add a loss term that holds them to it. The cost is
capacity and some accuracy; the benefit is that those dimensions can be read
directly, with no probe to fit and no probe to overfit.

CSS is a good first testbed. It is a real corpus written by humans at scale,
but its colour semantics are closed and exactly computable: `#a3f9c2` *is*
(163, 249, 194), and `hsl(240, 50%, 50%)` has an exact RGB equivalent. Ground
truth needs no annotation. And it contains several notations for the same
underlying quantity, which lets us ask whether a concept learned in one
notation transfers to another.

We ask three questions:

1. **Does anchoring work, and does it generalize?** Can dims 0/1/2 be made to
   carry RGB, and does that survive to colours never seen in training?
2. **Do unsupervised neighbours emerge?** `hsl()` is never anchored. Do
   dimensions take it on anyway?
3. **Is an emergent dimension's index meaningful?** If dimension 100 encodes
   hue in one run, does it encode hue in the next?

(3) is where we depart from the prior work this builds on, and it is the
question that decides what the first two findings are worth.

## 2. Method

### 2.1 Corpus

100,000 CSS files streamed from `bigcode/the-stack`, filtered to 100 B–1 MB,
at least five declarations (counted recursively through `@media` and friends),
parsing cleanly under `tinycss2`, and not machine-obfuscated. 594,854,006
tokens after tokenization.

### 2.2 Structural tokenization

A CSS-aware walker emits `(text, type, context)` triples, and the vocabulary is
a 70,004-entry "fat head" of bounded CSS vocabulary (named colours, properties,
units, pseudo-classes, function names) over a BPE tail for the rest.

The decision that matters: **hex codes are split into `#` plus three digit-pair
tokens**, with all 256 pairs `00`–`ff` atomic in the head. `#a3f9c2` becomes
`#`, `a3`, `f9`, `c2`.

This is not cosmetic. An earlier version of this pipeline fused each hex literal
into a single token; only the 4,704 most frequent made the head and the rest
were BPE-split at arbitrary boundaries. That model learned RGB *per specific hex
token* — a lookup table — and read unseen codes at r ≈ 0.03. The digit-pair
split changes what is learnable from a table into a function, and the 256³ ≈
16M combinations are all reachable through a fixed 256-token vocabulary.

The `context` tag also matters for correctness: `.red { }` must not anchor RGB
at a selector. Anchors fire only where `context == "value"`.

### 2.3 Architecture

GPT-2 decoder. 384 hidden dimensions, 6 layers, 6 heads, 1024-token context,
70,004 vocabulary, ~38M parameters.

> **A note on "1024."** That is the context length. The hidden state is 384-wide.
> The two get conflated easily and have been conflated in describing this work
> before; see [ERRATA.md](ERRATA.md).

### 2.4 The loss

```
L = L_LM  +  λ_anchor · L_anchor  +  λ_sparse · L_sparse
```

with `λ_anchor = 1.0`, `λ_sparse = 0.1`.

`L_anchor` is MSE between dims 0/1/2 and the true normalized RGB at anchored
positions. `L_sparse` is L2 on the remaining 381 dims at those same positions —
it discourages colour information from smearing across the rest of the state,
and is what makes the direct read meaningful rather than merely possible.

Anchors fire at three kinds of site:

| Site | Anchoring |
|---|---|
| hex digit-pairs | per channel: dim 0 at the R pair, dim 1 at G, dim 2 at B |
| named colours (`red`) | all three dims at the identifier |
| `rgb()` / `rgba()` | all three dims at the closing `)` |

27,575,097 RGB anchors in total. The per-channel hex scheme is the bulk of it
and produces three anchors per colour rather than one.

### 2.5 What is deliberately not supervised

**`hsl()` is never anchored.** The detector parses HSL values and returns them
in a separate list used only for later analysis; they never enter the anchor
set that the loss consumes:

```python
elif name in ("hsl", "hsla") and len(args) >= 3:
    hsls.append(HslDetection(close, name, h, s, l))   # analysis only
```

Everything reported in §3.3 rests on this line. No gradient ever asked any
dimension to represent hue, saturation or lightness.

### 2.6 Training

5000 steps, batch 4, AdamW (lr 3e-4, β = 0.9/0.95, weight decay 0.1), linear
warmup over 5% of steps then cosine decay, gradient clipping at 1.0, seed 42.
Roughly 10 hours on CPU. The anchor loss falls from ~0.99 to ~0.001.

## 3. Results

### 3.1 Supervised dimensions encode RGB, compositionally

Probing at synthesized uniformly-random hex codes, none of which appears as a
specific triple in training:

| Method | R | G | B | mean |
|---|---|---|---|---|
| **Direct read of dims 0/1/2** | **0.996** | **0.996** | **0.996** | **0.996** |
| Linear probe over all 384 dims | 0.999 | 0.999 | 0.998 | 0.999 |

Fitting a probe over the entire hidden state buys 0.003 Pearson over reading a
single dimension. The sparsity term did its job: there is essentially no RGB
information left anywhere else to recover.

Because the digit-pair vocabulary is fixed and atomic, this is a learned
function from (position of digit pair) to (channel value), and every one of the
~16M representable colours is reachable.

### 3.2 The structure is necessary, not decorative

An identical model — same architecture, same corpus, same compute — trained with
LM loss alone on a fresh 70K byte-level BPE, then probed with a fitted linear
map over all 384 dimensions:

| | R | G | B | mean | test R² |
|---|---|---|---|---|---|
| Anchored, direct read | 0.996 | 0.996 | 0.996 | **0.996** | – |
| LM-only, fitted probe | 0.551 | 0.285 | 0.654 | **0.497** | **−0.175** |

The baseline's G-channel test R² is **−0.739**: worse than predicting the
channel mean. Its BPE did tokenize hex characters individually, so the
information was present in the input. With only an LM objective, nothing ever
pushed it to compose those characters into a magnitude.

### 3.3 Unsupervised dimensions pick up HSL

Probing every non-supervised dimension against the true H, S and L of each
`hsl()` literal, at five candidate token positions. Hue is circular, so it is
projected onto sin/cos and the stronger basis is kept.

| Property | Best position | Dim | \|r\| |
|---|---|---|---|
| Saturation | S component token | 377 | **0.842** |
| Hue | H component token | 100 | 0.775 |
| Lightness | L component token | 69 | 0.774 |

Position matters a great deal. At the closing `)` — the single position the
original protocol specified — the same analysis gives S 0.635, H 0.535,
L 0.485. The information is most legible where it first becomes representable,
which is the component token itself, not the end of the call.

**Only saturation survives a stricter test.** Multivariate held-out R² at the
closing paren:

| sin(H) | cos(H) | S | L |
|---|---|---|---|
| −0.321 | 0.222 | **0.400** | 0.022 |

Hue is negative; lightness is indistinguishable from zero. The defensible
summary is: saturation crystallizes, hue and lightness show partial structure.

### 3.4 The index is not a fact about the model

This is the part that changes how the rest should be read.

**Multiple comparisons.** Taking the best of 381 dimensions is a selection
procedure, and against pure noise it does not return zero. We permute the target
values and re-run the identical argmax. At n = 250 examples the chance-level
peak is **|r| = 0.167** (p95 = 0.178). The floor falls roughly as 1/√n, so
sample size has to be reported alongside any correlation of this kind — a "weak
but present" |r| = 0.2 on a few hundred examples is indistinguishable from
nothing.

**Resampling moves the winner, with the model frozen.** Bootstrap-resampling
the evaluation examples and re-running the same argmax — same checkpoint, same
weights, same activations — the winning index moves. We observed this
accidentally first: at n = 250, hue's best dimension is 377; at n = 1500 it is
100. Same model, different sample, different answer.

⏳ *Bootstrap distribution over 500 resamples at n = 1500: pending.*

**Seeds move it too.** ⏳ *Three seeds, identical in every respect but
initialization and data order: pending. Predictions registered below.*

Two hypotheses make opposite predictions, and they are not the same claim:

| | Strong / index | Weak / count |
|---|---|---|
| Claim | Dimension 100 encodes hue | *Some* small set encodes hue at \|r\| ≈ 0.8 |
| Best dim across seeds | recurs | scatters |
| Top-k overlap vs chance | far above | at chance |
| Best \|r\| across seeds | stable | **also stable** |
| Count above threshold | stable | **also stable** |

Note the bottom two rows agree. Magnitude and count cannot discriminate the
hypotheses; only index agreement and set overlap can. A paper that reports only
"we found dimension 100 encodes hue at 0.78" has not tested the thing it sounds
like it tested.

We expect the weak reading. The evidence already in hand — resampling alone
moves the index — makes the strong reading hard to sustain.

### 3.5 Nothing propagates across formats

Read the supervised R/G/B dims at `hsl()` positions and compare against that
colour's true RGB: |r| = 0.06–0.36 across all five probe positions, against a
shuffled baseline of 0.02.

The model never learned to convert HSL to RGB, and there is a clean reason.
The anchor loss fires only at anchored positions. The token following
`hsl(240, 50%, 50%)` in real CSS is almost always `;`, `}` or `!important`.
Predicting `;` does not require knowing the colour. No gradient pressure, no
propagation — a property of the objective, not a tuning failure.

## 4. Discussion

**Supervision buys an address; emergence buys capacity.** The anchored
dimensions are readable, compositional, and stable by construction — dimension 0
is red because the loss says so, in every run. The emergent dimensions genuinely
encode something real, but at coordinates that are an accident of the
optimization. Both are true at once, and conflating them is the trap.

For interpretability-by-construction the lesson is unglamorous and actionable:
**every dimension you want to read, you must name and supervise.** You cannot
train a model, find that some dimension correlates with a concept, and treat
that index as an interface. It will not survive a reseed — and, at small
evaluation sets, may not survive a resample.

**The emergent result is still worth something.** That ~40–120 dimensions reach
|r| ≥ 0.5 for saturation, unprompted, says the representation has spare capacity
organizing itself around colour structure the loss never mentioned. That is a
claim about *how much* structure is latently available to be named, which is
useful for deciding what to spend a dimensional budget on. It is not a claim
about *where*.

**Two conditions for a named dimension to work**, both necessary:

1. *The input tokenization must carry compositional structure for the concept.*
   Fused hex tokens gave r ≈ 0.03; digit pairs gave 0.996. Same loss, same
   architecture.
2. *The loss must fire at every position where you want the dimension to mean
   something.* It fires at hex and named colours, and those read cleanly. It
   never fires at `hsl()`, and RGB is absent there.

The original protocol's negative result on HSL was caused by failing (1) and by
probing at the wrong position simultaneously. Both look like "the method
doesn't work" from the outside.

## 5. Limitations

- **One domain, and an unusually clean one.** CSS colour has closed, computable
  semantics. Nothing here shows the approach survives contact with concepts that
  lack ground truth.
- **Small model.** 38M parameters, 384 dimensions. Whether the sparsity term
  scales, or whether reserved dimensions become unaffordable at frontier width,
  is untested.
- **Three dimensions of ~384.** The interpretability cost of naming a large
  fraction of the state is unmeasured.
- **LM quality was monitored, not optimized.** Loss falls 11.18 → ~3.2 and
  generated samples are structurally plausible (77% contain `{}`, 97% contain
  selectors), but we did not measure what the anchor and sparsity terms cost in
  perplexity against a matched unconstrained run.
- **Correlational probes.** Nothing here is causal. Showing that dimension 377
  *carries* saturation is not showing the model *uses* it. Ablation or patching
  would be the next step.

## 6. Reproducibility

Code, result artifacts and exact commands: [REPRODUCE.md](REPRODUCE.md).
Corrections to the previously published description of this experiment:
[ERRATA.md](ERRATA.md).

The checkpoint behind every number in §3.1–3.3 is published, so the analyses
can be re-run in minutes without repeating the training.
