# Reproduction record

## 2026-09-18 — HSL emergence reproduces exactly from the checkpoint

`probes/stage8c_multiprobe.py` was re-run from scratch against
`data/parsed/stage4/checkpoints/step_005000.pt` and its output compared
element-by-element against the result JSON committed when the experiment was
originally run (May 2026).

```bash
python probes/stage8c_multiprobe.py --n-files 20000 --max-examples 1500
```

1500 examples, same as the original run. **Every best-dimension index is
identical and every correlation matches to ~1e-14** — floating-point
serialization noise, nothing more.

| Probe position | Prop | Dim (orig → rerun) | \|r\| (orig) | \|r\| (rerun) | Δ |
|---|---|---|---|---|---|
| `hsl(` opener | H | 43 → 43 | 0.449035089 | 0.449035089 | 6.1e-15 |
| `hsl(` opener | S | 50 → 50 | 0.513952003 | 0.513952003 | 2.4e-15 |
| `hsl(` opener | L | 333 → 333 | 0.465987531 | 0.465987531 | 4.4e-15 |
| **H component** | **H** | **100 → 100** | **0.774561344** | **0.774561344** | 1.0e-14 |
| H component | S | 169 → 169 | 0.730816404 | 0.730816404 | 5.1e-15 |
| H component | L | 42 → 42 | 0.492894719 | 0.492894719 | 5.4e-15 |
| S component | H | 121 → 121 | 0.655833255 | 0.655833255 | 9.8e-15 |
| **S component** | **S** | **377 → 377** | **0.841995747** | **0.841995747** | 3.3e-15 |
| S component | L | 305 → 305 | 0.552369074 | 0.552369074 | 7.1e-15 |
| L component | H | 348 → 348 | 0.479177855 | 0.479177855 | 8.0e-15 |
| L component | S | 42 → 42 | 0.548896239 | 0.548896239 | 2.4e-15 |
| **L component** | **L** | **69 → 69** | **0.773520257** | **0.773520257** | 7.2e-15 |
| `)` baseline | H | 176 → 176 | 0.535076337 | 0.535076337 | 7.2e-15 |
| `)` baseline | S | 176 → 176 | 0.634669903 | 0.634669903 | 2.6e-15 |
| `)` baseline | L | 249 → 249 | 0.484622710 | 0.484622710 | 6.2e-15 |

The headline three — **377 saturation, 100 hue, 69 lightness** — are confirmed.

The negative result also reproduced: no probe position shows dims 0/1/2 firing
with the true RGB of the HSL colour. Per-probe |r| ranged −0.399 to +0.405,
mean |r| ≈ 0.25.

### What this does and does not establish

It establishes that **the published numbers follow from the published
checkpoint and the published code**. Given the same weights and the same
corpus, the analysis is deterministic and the reported values are what the code
actually emits.

It does **not** establish that a retrained model would land on 377/100/69.
That is a different question, and the one `probes/seed_robustness.py` exists to
answer. Early evidence says it would not — see below.

## 2026-09-18 — index instability observed under resampling alone

While wiring up the robustness probe on a smaller sample (n=250), the winning
dimensions came out **different from the n=1500 run on the same checkpoint**.
Both columns below are `step_005000.pt`, each property read at its own
component-token position:

| Property | n=250 | n=1500 |
|---|---|---|
| Hue | dim 377 | dim 100 |
| Lightness | dim 22 | dim 69 |
| Saturation | dim 377 | dim 377 |

Same weights, same code, same corpus — only the number of HSL examples drawn
differs. The argmax over 381 dimensions is not stable to the evaluation sample.

The shuffled-target control on that run put the chance-level peak at
**|r| = 0.167** (n=250, p95 = 0.178), which is not a small number. Taking the
best of 381 dimensions is a selection procedure and its floor has to be
reported alongside the result.

This motivated adding a formal bootstrap test to the probe, rather than leaving
seed variation as the only robustness check. Results in
`bootstrap_stability.json`.
