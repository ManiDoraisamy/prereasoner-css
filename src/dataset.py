"""
Memory-mapped, windowed CSS dataset.

Reads the binary blob produced by stage4_preencode.py and yields fixed-length
context windows along with the color anchors that fall inside each window.
Anchors carry positions in *ID-space within their file*; we translate them to
window-local positions on item access.

Window strategy: per-file non-overlapping windows of length context_len.
A trailing window shorter than context_len is kept iff its length >=
min_window_len (default context_len // 2). Windows never cross file
boundaries — the model never sees two unrelated CSS files glued together.

The dataset is RAM-light: the four binary files are mmap'd, only the small
window-index list is held in Python. Works for 1M-file corpora as easily as
for the 100k development set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ANCHOR_DTYPE = np.dtype([
    ("file_idx", np.uint32),
    ("pos", np.uint32),
    ("r", np.float32),
    ("g", np.float32),
    ("b", np.float32),
    ("channel", np.int8),
    ("_pad", np.uint8, 3),
])


class MmapWindowedDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        context_len: int = 1024,
        max_files: int | None = None,
        min_window_len: int | None = None,
        max_windows: int | None = None,
    ):
        d = Path(data_dir)
        self.tokens = np.memmap(d / "tokens.bin", dtype=np.uint32, mode="r")
        self.file_offsets = np.memmap(d / "file_offsets.bin", dtype=np.uint64, mode="r")
        self.anchors = np.memmap(d / "anchors.bin", dtype=ANCHOR_DTYPE, mode="r")
        self.anchor_offsets = np.memmap(
            d / "anchor_offsets.bin", dtype=np.uint64, mode="r",
        )

        with (d / "meta.json").open() as f:
            self.meta = json.load(f)

        self.context_len = context_len
        self.min_window_len = min_window_len if min_window_len is not None else context_len // 2

        n_files_total = len(self.file_offsets) - 1
        n_files = min(n_files_total, max_files) if max_files else n_files_total

        # Pre-build the window index: (file_idx, start_in_file, length)
        # Stored as three parallel uint32 arrays to keep RAM tiny.
        win_f: list[int] = []
        win_s: list[int] = []
        win_l: list[int] = []
        for f in range(n_files):
            file_len = int(self.file_offsets[f + 1] - self.file_offsets[f])
            if file_len == 0:
                continue
            if file_len <= context_len:
                win_f.append(f); win_s.append(0); win_l.append(file_len)
                continue
            n_full = file_len // context_len
            for w in range(n_full):
                win_f.append(f); win_s.append(w * context_len); win_l.append(context_len)
            tail = file_len - n_full * context_len
            if tail >= self.min_window_len:
                win_f.append(f); win_s.append(n_full * context_len); win_l.append(tail)
            if max_windows is not None and len(win_f) >= max_windows:
                break

        if max_windows is not None and len(win_f) > max_windows:
            win_f = win_f[:max_windows]; win_s = win_s[:max_windows]; win_l = win_l[:max_windows]

        self.win_file = np.asarray(win_f, dtype=np.uint32)
        self.win_start = np.asarray(win_s, dtype=np.uint32)
        self.win_len = np.asarray(win_l, dtype=np.uint32)

    def __len__(self) -> int:
        return int(self.win_file.size)

    def __getitem__(self, idx: int) -> dict:
        file_idx = int(self.win_file[idx])
        start = int(self.win_start[idx])
        length = int(self.win_len[idx])

        token_start = int(self.file_offsets[file_idx]) + start
        token_end = token_start + length
        # int64 because PyTorch indices want int64; uint32 in storage is fine
        ids = np.asarray(self.tokens[token_start:token_end], dtype=np.int64)

        a_start = int(self.anchor_offsets[file_idx])
        a_end = int(self.anchor_offsets[file_idx + 1])
        if a_end > a_start:
            file_anchors = self.anchors[a_start:a_end]
            mask = (file_anchors["pos"] >= start) & (file_anchors["pos"] < start + length)
            in_window = file_anchors[mask]
            anchor_pos = in_window["pos"].astype(np.int64) - start
            anchor_rgb = np.stack(
                [in_window["r"], in_window["g"], in_window["b"]], axis=1,
            ).astype(np.float32)
            anchor_channel = in_window["channel"].astype(np.int64)
        else:
            anchor_pos = np.zeros(0, dtype=np.int64)
            anchor_rgb = np.zeros((0, 3), dtype=np.float32)
            anchor_channel = np.zeros(0, dtype=np.int64)

        return {
            "input_ids": torch.from_numpy(ids),
            "anchor_pos": torch.from_numpy(anchor_pos),
            "anchor_rgb": torch.from_numpy(anchor_rgb),
            "anchor_channel": torch.from_numpy(anchor_channel),
        }


def collate_with_anchors(batch: list[dict], pad_id: int) -> dict:
    """Pad input_ids to the batch max length, merge per-sequence anchors into
    batch-indexed flat tensors."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)

    batch_idx_list: list[int] = []
    pos_list: list[int] = []
    rgb_list: list[torch.Tensor] = []
    channel_list: list[torch.Tensor] = []

    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = 1
        a_pos = item["anchor_pos"]
        a_rgb = item["anchor_rgb"]
        a_ch = item["anchor_channel"]
        if a_pos.numel() > 0:
            batch_idx_list.extend([i] * a_pos.numel())
            pos_list.extend(a_pos.tolist())
            rgb_list.append(a_rgb)
            channel_list.append(a_ch)

    if rgb_list:
        anchor_rgb = torch.cat(rgb_list, dim=0)
        anchor_channel = torch.cat(channel_list, dim=0)
    else:
        anchor_rgb = torch.zeros((0, 3), dtype=torch.float32)
        anchor_channel = torch.zeros((0,), dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "anchor_batch": torch.tensor(batch_idx_list, dtype=torch.long),
        "anchor_pos": torch.tensor(pos_list, dtype=torch.long),
        "anchor_rgb": anchor_rgb,
        "anchor_channel": anchor_channel,
    }
