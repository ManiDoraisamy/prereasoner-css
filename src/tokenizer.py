"""
CSSTokenizer: unified whole-token fat-head + BPE rare-tail tokenizer.

Unified ID space:
  - IDs [0, fat_size)              -> fat head (incl. specials)
  - IDs [fat_size, fat_size + B)   -> BPE sub-words, where B = bpe vocab size

Usage:
  tok = CSSTokenizer.load("data/parsed/stage2")
  ids = tok.encode_tokens(walk_css(css_text))   # from a token stream
  ids = tok.encode_css(css_text)                # parse + encode in one shot
  tokens = tok.decode(ids)                      # returns list[(text, type)]
  css = serialize_tokens(tokens)                # see css_walker.serialize_tokens
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer

from css_walker import walk_css, BPE_TYPES, T_STRING, T_URL


class CSSTokenizer:
    def __init__(
        self,
        fat_head: dict[str, int],
        fat_head_meta: dict[str, dict],
        bpe: Tokenizer,
        config: dict,
    ):
        self.fat_head = fat_head
        self.fat_head_meta = fat_head_meta
        self.bpe = bpe
        self.config = config

        self.fat_size = config["fat_head_size"]
        self.bpe_offset = config["bpe_id_offset"]
        self.total_size = config["total_vocab_size"]
        self.bpe_routed_types = set(config["bpe_routed_types"])

        # Reverse map: id -> text (fat head only; BPE handles its own decoding)
        self.fat_head_inv = {v: k for k, v in fat_head.items()}

        # Specials
        self.pad_id = fat_head.get("<pad>")
        self.bos_id = fat_head.get("<bos>")
        self.eos_id = fat_head.get("<eos>")
        self.unk_id = fat_head.get("<unk>")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, dir_path: str | Path) -> "CSSTokenizer":
        d = Path(dir_path)
        with (d / "fat_head.json").open(encoding="utf-8") as f:
            fat_head = json.load(f)
        with (d / "fat_head_meta.json").open(encoding="utf-8") as f:
            fat_head_meta = json.load(f)
        with (d / "tokenizer_config.json").open(encoding="utf-8") as f:
            config = json.load(f)
        bpe = Tokenizer.from_file(str(d / "bpe.json"))
        return cls(fat_head, fat_head_meta, bpe, config)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_token(self, text: str, ttype: str) -> list[int]:
        """Encode a single (text, type) token into one or more IDs."""
        # Strings and URLs always route to BPE (unbounded vocabulary)
        if ttype not in self.bpe_routed_types:
            fid = self.fat_head.get(text)
            if fid is not None:
                return [fid]
        # BPE path
        enc = self.bpe.encode(text)
        return [bpe_id + self.bpe_offset for bpe_id in enc.ids]

    def encode_tokens(
        self,
        tokens: Iterable,
        add_special: bool = False,
    ) -> list[int]:
        """Encode an iterable of (text, type) or (text, type, context) tokens."""
        out: list[int] = []
        if add_special and self.bos_id is not None:
            out.append(self.bos_id)
        for tok in tokens:
            text, ttype = tok[0], tok[1]
            out.extend(self.encode_token(text, ttype))
        if add_special and self.eos_id is not None:
            out.append(self.eos_id)
        return out

    def encode_tokens_with_positions(
        self,
        tokens: list,
        add_special: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Encode and return (ids, stream_to_id_start).

        stream_to_id_start[i] gives the position in `ids` where stream token i
        begins. Critical for mapping color-anchor positions (which are computed
        on the token stream) to the ID-position space the model sees.
        Accepts (text, type) or (text, type, context) tokens.
        """
        ids: list[int] = []
        starts: list[int] = []
        if add_special and self.bos_id is not None:
            ids.append(self.bos_id)
        for tok in tokens:
            text, ttype = tok[0], tok[1]
            starts.append(len(ids))
            ids.extend(self.encode_token(text, ttype))
        if add_special and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids, starts

    def encode_css(self, css_text: str, add_special: bool = False) -> list[int]:
        return self.encode_tokens(walk_css(css_text), add_special=add_special)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> list[tuple[str, str]]:
        """Decode IDs back to (text, type) tokens.

        Contiguous BPE IDs are merged into a single sub-word string. The
        node type for BPE-decoded tokens is set to "bpe" since we don't
        store the original type or context after encoding.
        """
        out: list[tuple[str, str]] = []
        bpe_buf: list[int] = []

        def flush_bpe():
            if not bpe_buf:
                return
            local_ids = [i - self.bpe_offset for i in bpe_buf]
            text = self.bpe.decode(local_ids)
            out.append((text, "bpe"))
            bpe_buf.clear()

        for tid in ids:
            if tid >= self.fat_size:
                bpe_buf.append(tid)
            else:
                flush_bpe()
                text = self.fat_head_inv.get(tid, "<unk>")
                meta = self.fat_head_meta.get(text, {})
                ttype = meta.get("primary_type", "ident")
                out.append((text, ttype))
        flush_bpe()
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"CSSTokenizer(fat_head={self.fat_size:,}, "
            f"bpe={self.config['bpe_vocab_size']:,}, "
            f"total={self.total_size:,})"
        )
