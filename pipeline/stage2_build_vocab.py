"""
Stage 2.2: Build the unified tokenizer vocabulary.

Reads:  data/parsed/stage2/token_freq.json  (from stage2_extract_tokens.py)
Writes: data/parsed/stage2/fat_head.json        {token: id}
        data/parsed/stage2/fat_head_meta.json   per-token metadata (count, primary type)
        data/parsed/stage2/bpe.json             HuggingFace tokenizers BPE artifact
        data/parsed/stage2/tokenizer_config.json combined metadata

The fat head holds the top N most-frequent tokens (excluding strings/URLs,
which are unbounded and always BPE-encoded). Everything else trains a BPE
tokenizer with the configured rare-tail vocab size.

Usage:
  python stage2_build_vocab.py
  python stage2_build_vocab.py --fat-head 50000 --bpe-vocab 20000
"""

# --- repo path shim: make src/ importable without installing a package ---
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "src"))
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
# --- end shim ---

import argparse
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from css_force_include import ALL_FORCE_INCLUDE, CATEGORY_SIZES
from css_walker import BPE_TYPES

OUT_DIR = Path("data/parsed/stage2")
FREQ_PATH = OUT_DIR / "token_freq.json"
STREAMS_PATH = OUT_DIR / "token_streams.jsonl"

# Special tokens reserved at the bottom of the fat head ID range
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fat-head", type=int, default=50_000,
                        help="Fat-head vocabulary size (default 50k)")
    parser.add_argument("--bpe-vocab", type=int, default=20_000,
                        help="BPE vocabulary size for the rare tail (default 20k)")
    args = parser.parse_args()

    if not FREQ_PATH.exists():
        print(f"Missing {FREQ_PATH}. Run stage2_extract_tokens.py first.")
        return

    print(f"Loading frequency table from {FREQ_PATH}...")
    with FREQ_PATH.open(encoding="utf-8") as f:
        freq = json.load(f)
    print(f"  {len(freq):,} unique tokens loaded")

    # Partition: BPE-only types skip fat-head entirely
    fat_candidates = []
    rare_tail_seed: Counter = Counter()

    for tok, v in freq.items():
        types = v["types"]
        primary_type = max(types.items(), key=lambda x: x[1])[0]
        if primary_type in BPE_TYPES:
            # Always route to BPE
            rare_tail_seed[tok] += v["count"]
        else:
            fat_candidates.append((tok, v["count"], primary_type))

    # Sort fat candidates by count
    fat_candidates.sort(key=lambda x: -x[1])

    # Force-include pass: reserve fat-head slots for bounded-vocabulary CSS
    # tokens regardless of their corpus frequency. These come from the spec
    # (named colors, property names, units, function names, etc.) and must
    # stay atomic so BPE never splits mid-symbol.
    candidate_dict = {tok: (count, primary_type) for tok, count, primary_type in fat_candidates}
    forced_already_present: list[str] = []
    forced_added: list[str] = []
    for token in ALL_FORCE_INCLUDE:
        if token in candidate_dict:
            forced_already_present.append(token)
        else:
            forced_added.append(token)

    # Top frequency tokens up to (fat_head - forced_added count) — but we still
    # want all forced_already_present tokens in the fat head even if they fall
    # below the cutoff.
    needed_slots_for_forced = len(forced_added) + sum(
        1 for t in forced_already_present
        if t not in {x[0] for x in fat_candidates[: args.fat_head - len(forced_added)]}
    )
    freq_slots = max(0, args.fat_head - needed_slots_for_forced)
    by_freq = fat_candidates[:freq_slots]
    seen_tokens = {tok for tok, _, _ in by_freq}

    # Add forced-already-present that didn't make the freq slice
    extra_forced_present: list[tuple[str, int, str]] = []
    for t in forced_already_present:
        if t not in seen_tokens:
            count, primary_type = candidate_dict[t]
            extra_forced_present.append((t, count, primary_type))
            seen_tokens.add(t)

    # Add forced-added (zero-count, marked 'forced')
    added_recs = [(t, 0, "forced") for t in forced_added if t not in seen_tokens]
    for t, _, _ in added_recs:
        seen_tokens.add(t)

    fat_head_tokens = list(by_freq) + extra_forced_present + added_recs

    # Cut to fat_head budget if we somehow overshot
    fat_head_tokens = fat_head_tokens[: args.fat_head]
    final_tokens = {tok for tok, _, _ in fat_head_tokens}

    # Anything cut from the fat head feeds BPE training (so unusual idents
    # can still be encoded)
    for tok, count, _ in fat_candidates:
        if tok not in final_tokens:
            rare_tail_seed[tok] += count

    print(f"\nFat head : {len(fat_head_tokens):,} tokens")
    print(f"  forced-include categories: {CATEGORY_SIZES}")
    print(f"  forced already in fat head by frequency : {len(forced_already_present):,}")
    print(f"  forced added on top                     : {len(forced_added):,}")
    print(f"Rare tail seed for BPE : {len(rare_tail_seed):,} unique tokens")

    # Build fat head mapping (specials first)
    fat_head_map: dict[str, int] = {}
    fat_head_meta: dict[str, dict] = {}
    for i, sp in enumerate(SPECIAL_TOKENS):
        fat_head_map[sp] = i
        fat_head_meta[sp] = {"count": 0, "primary_type": "special"}

    next_id = len(SPECIAL_TOKENS)
    for tok, count, primary_type in fat_head_tokens:
        if tok in fat_head_map:
            continue  # collision with a special token name
        fat_head_map[tok] = next_id
        fat_head_meta[tok] = {"count": count, "primary_type": primary_type}
        next_id += 1

    fat_path = OUT_DIR / "fat_head.json"
    meta_path = OUT_DIR / "fat_head_meta.json"
    with fat_path.open("w", encoding="utf-8") as f:
        json.dump(fat_head_map, f, ensure_ascii=False)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(fat_head_meta, f, ensure_ascii=False)
    print(f"Fat head saved to {fat_path} (max id = {next_id - 1})")

    # ------------------------------------------------------------------
    # Train BPE on rare tail
    # ------------------------------------------------------------------
    print(f"\nTraining BPE on rare tail (target vocab {args.bpe_vocab:,})...")

    # Build a training corpus by repeating each token by its count, capped to
    # avoid blowing up memory on hot tokens.
    CAP = 100  # max repetitions per unique token
    training_lines = []
    for tok, count in rare_tail_seed.items():
        reps = min(count, CAP)
        training_lines.extend([tok] * reps)
    print(f"  Training corpus: {len(training_lines):,} lines")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.bpe_vocab,
        special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(training_lines, trainer=trainer)

    bpe_path = OUT_DIR / "bpe.json"
    tokenizer.save(str(bpe_path))
    print(f"BPE saved to {bpe_path}")

    # ------------------------------------------------------------------
    # Combined config
    # ------------------------------------------------------------------
    fat_size = len(fat_head_map)
    bpe_size = tokenizer.get_vocab_size()
    total = fat_size + bpe_size

    config = {
        "fat_head_size": fat_size,
        "bpe_vocab_size": bpe_size,
        "total_vocab_size": total,
        "specials": SPECIAL_TOKENS,
        "bpe_id_offset": fat_size,  # BPE IDs are shifted by fat_size in the unified space
        "bpe_routed_types": sorted(BPE_TYPES),
    }
    cfg_path = OUT_DIR / "tokenizer_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {cfg_path}")

    print(f"\n--- Stage 2 vocab build complete ---")
    print(f"Fat head size       : {fat_size:,}")
    print(f"BPE vocab size      : {bpe_size:,}")
    print(f"Total unified vocab : {total:,}")


if __name__ == "__main__":
    main()
