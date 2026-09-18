# Publishing the checkpoint

The trained model is 435 MB — too large for git, and the thing a reader needs
in order to check the numbers without spending ten hours retraining. Hosting it
turns tier-1 reproduction from "rebuild the corpus first" into "download and
run".

## What to upload

| File | Size | Why |
|---|---|---|
| `data/parsed/stage4/checkpoints/step_005000.pt` | 435 MB | the trained model |
| `data/parsed/stage2/fat_head.json` | ~2 MB | vocabulary — the checkpoint is unreadable without it |
| `data/parsed/stage2/bpe.json` | ~1 MB | rare-token tail |
| `data/parsed/stage2/tokenizer_config.json` | small | |
| `data/parsed/stage2/fat_head_meta.json` | small | |

The tokenizer is not optional. Token IDs are meaningless without the exact
vocabulary that produced them, and a mismatched `fat_head.json` will silently
produce garbage rather than an error.

**Do not upload** `data/stage1/` (the CSS corpus). It is redistributed
source code from other people's repositories; `pipeline/stage1_download.py`
rebuilds it from `bigcode/the-stack` under that dataset's own terms.

## Upload

```bash
pip install huggingface_hub
huggingface-cli login

python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = "<user>/prereasoner-css-phase0"

api.create_repo(repo, repo_type="model", exist_ok=True, private=False)

api.upload_file(
    path_or_fileobj="data/parsed/stage4/checkpoints/step_005000.pt",
    path_in_repo="step_005000.pt",
    repo_id=repo, repo_type="model",
)
api.upload_folder(
    folder_path="data/parsed/stage2",
    path_in_repo="tokenizer",
    repo_id=repo, repo_type="model",
)
PY
```

## Download, for a reader

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
import shutil

repo = "<user>/prereasoner-css-phase0"
Path("data/parsed/stage4/checkpoints").mkdir(parents=True, exist_ok=True)

shutil.copy(
    hf_hub_download(repo, "step_005000.pt"),
    "data/parsed/stage4/checkpoints/step_005000.pt",
)
snapshot_download(repo, allow_patterns="tokenizer/*", local_dir="data/_hf")
shutil.copytree("data/_hf/tokenizer", "data/parsed/stage2", dirs_exist_ok=True)
PY
```

Then the analyses in [REPRODUCE.md](REPRODUCE.md) tier 1 run directly. Note
that `stage8c_multiprobe.py` also needs the CSS corpus to draw HSL examples
from, so run `pipeline/stage1_download.py` first — or point `--css-dir` at any
directory of CSS files, accepting that the examples will differ.

## Suggested model card

```markdown
---
license: mit
tags: [interpretability, css, named-dimensions, mechanistic-interpretability]
---

# prereasoner-css-phase0

38M-parameter GPT-2 trained on 100k CSS files with hidden dimensions 0, 1 and 2
constrained by an auxiliary MSE loss to carry the R, G and B channels of every
colour literal in the text.

- 384 hidden dims, 6 layers, 6 heads, 1024 context, 70,004 vocab
- 5000 steps, batch 4, AdamW lr 3e-4, seed 42
- Dims 0/1/2 read RGB off unseen hex codes at r = 0.996, directly, no probe

`hsl()` is never anchored, yet dims 377 / 100 / 69 come to encode saturation /
hue / lightness at |r| = 0.84 / 0.78 / 0.77. **Those indices are specific to
this checkpoint** and do not survive reseeding — see the repo for the
robustness analysis.

Code, analyses and caveats: https://github.com/<user>/prereasoner-css
```

## Before you push

- [ ] `.env` is gitignored and no `HF_TOKEN` appears in any tracked file
- [ ] The checkpoint loads standalone: `torch.load(..., weights_only=False)` has
      the `config` and `model_state` keys the probes expect
- [ ] `hidden_dim` in the checkpoint config reads 384 — if something says 1024,
      that is the context length, and it is the exact confusion
      [ERRATA.md](ERRATA.md) exists to correct
- [ ] The model card says the HSL indices are checkpoint-specific. This is the
      claim most likely to be repeated without its caveat.
