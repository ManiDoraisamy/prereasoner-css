# Reproducing the results

Three tiers. Pick by how much you want to take on faith.

| Tier | What you verify | Cost |
|---|---|---|
| 1 | The published numbers follow from the published checkpoint | ~20 min |
| 2 | The checkpoint follows from the corpus | ~10 h CPU / ~1 h A100 |
| 3 | The finding is a property of the method, not of one seed | ~10 h per seed |

Everything below runs from the repo root. `data/` is gitignored; the layout
the scripts expect is:

```
data/
  stage1/css/              100k .css files
  stage1/meta/             100k .json sidecars
  parsed/stage2/           tokenizer (fat_head.json, bpe.json, ...)
  parsed/stage4/           tokens.bin, anchors.bin, *_offsets.bin, meta.json
  parsed/stage4/checkpoints/step_005000.pt
  parsed/stage8/           result JSONs
  seeds/seed<N>/           multi-seed checkpoints
```

---

## Tier 1 — verify the numbers from the checkpoint

You need `data/parsed/stage2/` (tokenizer, ~3 MB), `data/stage1/css/`
(corpus, ~1.6 GB) and the checkpoint (435 MB).

```bash
pip install -r requirements.txt

# HSL emergence — the headline result
python probes/stage8c_multiprobe.py \
    --ckpt data/parsed/stage4/checkpoints/step_005000.pt \
    --n-files 20000 --max-examples 1500

# RGB on hex codes never seen in training
python probes/stage8d_hex_generalization.py

# The negative result: RGB does not propagate to HSL positions
python probes/stage8b_rgb_propagation.py
```

Expected from `stage8c_multiprobe.py`:

```
  S component token    dim 377   |r| = 0.842
  H component token    dim 100   |r| = 0.775
  L component token    dim  69   |r| = 0.774
```

`hsl()` is sparse in real CSS — roughly one usable example per 4–5 files. With
`--n-files` below ~1500 you will not reach the 100-example floor and the script
will refuse to report. This is deliberate.

## Tier 2 — retrain from the corpus

Needs a HuggingFace token with access to `bigcode/the-stack`. Put it in
`.env` as `HF_TOKEN=...` (gitignored; never commit it).

```bash
python pipeline/stage1_download.py --target 100000
python pipeline/stage1_verify.py            # expect ~92% fully clean
python pipeline/stage2_extract_tokens.py
python pipeline/stage2_build_vocab.py
python pipeline/stage2_test_roundtrip.py    # expect 100% round-trip
python pipeline/stage4_preencode.py
python pipeline/stage4_train.py --steps 5000 --batch-size 4
python pipeline/stage5_validate.py
```

Download is the long pole — it streams and filters the CSS shard of the-stack.
Filters: 100 B–1 MB, ≥5 declarations (counted recursively through `@media`),
parses under tinycss2, not obfuscated.

Checkpoints go to `data/parsed/stage4/checkpoints/`. The anchor loss should
fall from ~0.99 to ~0.001; RGB MAE should reach ~0.03 per channel.

**Expect different dimension indices than 377/100/69.** A fresh run is a fresh
draw. That is the subject of tier 3, not a sign anything is broken.

## Tier 3 — the multi-seed study

```bash
# Check the wiring first — minutes, not hours.
python pipeline/run_seeds.py --seeds 42 43 44 --smoke --steps 200

# The real sweep. Sequential; budget ~10 h per seed on CPU.
python pipeline/run_seeds.py --seeds 42 43 44 --steps 5000 --batch-size 4 --skip-existing

python probes/seed_robustness.py \
    --ckpt data/seeds/seed42/step_005000.pt \
    --ckpt data/seeds/seed43/step_005000.pt \
    --ckpt data/seeds/seed44/step_005000.pt \
    --out results/seed_robustness.json
```

`run_seeds.py` writes `data/seeds/manifest.json` after every seed, so a crash
mid-sweep leaves a record and `--skip-existing` resumes.

The seed probe reports four things:

- **index agreement** — how often two seeds pick the same best dim. Under the
  weak/count reading this should be near zero.
- **top-k overlap** — mean pairwise Jaccard of top-k dim sets, printed next to
  the chance rate for k of 381. Overlap at chance means the indices carry no
  information across runs.
- **magnitude stability** — mean ± std of the best |r|. Under the weak/count
  reading this should be tight even as the indices scatter.
- **shuffled-target control** — the chance-level peak |r| when the target is
  permuted and the same argmax-over-381-dims is run. This is the floor a real
  correlation has to clear, and it is not zero.

---

## Getting the artifacts

**Small result JSONs** are tracked in `results/` — a few hundred KB. They are
enough to check the reported numbers against the code that emits them.

**The corpus** is not redistributed; `pipeline/stage1_download.py` rebuilds it
from the-stack. The filters are deterministic, but the-stack is a moving
target, so a rebuild today may not be byte-identical to the original 100k.

**The checkpoint** (435 MB) is too large for git. See
[docs/PUBLISH_CHECKPOINT.md](PUBLISH_CHECKPOINT.md) for the upload/download
path.

---

## Known gotchas

- **Run from the repo root.** Data paths are relative to it. The entry scripts
  carry a `sys.path` shim for `src/`, so `python probes/foo.py` works from the
  root without installing anything.
- **`hidden_dim` is 384, not 1024.** 1024 is `context_len`. This trips people
  up; it tripped up the essay. See [ERRATA.md](ERRATA.md).
- **Dims 0/1/2 are excluded from every emergent-dim search.** They are
  supervised, so including them would just rediscover the anchor.
- **Hue is circular.** Correlating against raw degrees is meaningless — 359° and
  1° are neighbours. The probes project onto sin/cos and keep the stronger
  basis.
- **CPU only in the reference runs.** `torch` here is a CPU build; the timings
  quoted are 16-core CPU. A CUDA build cuts training to about an hour.
