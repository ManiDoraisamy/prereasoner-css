"""
NamedDimModel: GPT-2 style decoder with three designated embedding dimensions
(R, G, B at indices 0, 1, 2) and optional per-position activation logging.

The "naming" is purely an index reservation; no architectural change is needed.
Stage 4's constraint loss is what actually anchors these dimensions to RGB
values at color positions. Until then, dims 0/1/2 train like any other.

Activation logging captures the final-layer hidden states at every token
position. Use this during Stage 8 discovery (or any inference-time analysis)
to map activations back to source tokens for correlation studies.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


# ---------------------------------------------------------------------------
# Toy-scale default config (Stage 3-5). Production config (Stage 7) overrides.
# ---------------------------------------------------------------------------

@dataclass
class NamedDimConfig:
    vocab_size: int = 70_004
    hidden_dim: int = 384        # n_embd
    n_layers: int = 6            # n_layer
    n_heads: int = 6             # n_head
    context_len: int = 1024      # n_positions
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    named_dims: dict[str, int] = field(default_factory=lambda: {"R": 0, "G": 1, "B": 2})


# ---------------------------------------------------------------------------
# Activation logger
# ---------------------------------------------------------------------------

class ActivationLogger:
    """Append per-position hidden states to an in-memory buffer; flush to .npz."""

    def __init__(self, out_path: str | Path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._batches: list[np.ndarray] = []   # each (batch, seq, hidden)
        self._token_ids: list[np.ndarray] = [] # each (batch, seq)

    def log(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> None:
        # detach + cpu + float32 for portable storage
        self._batches.append(hidden_states.detach().cpu().float().numpy())
        self._token_ids.append(input_ids.detach().cpu().numpy())

    def flush(self) -> None:
        if not self._batches:
            return
        # Concatenate along batch dimension only if shapes match;
        # otherwise save as object array.
        try:
            stacked_hidden = np.concatenate(self._batches, axis=0)
            stacked_ids = np.concatenate(self._token_ids, axis=0)
            np.savez_compressed(self.out_path,
                                hidden_states=stacked_hidden,
                                input_ids=stacked_ids)
        except ValueError:
            np.savez_compressed(
                self.out_path,
                hidden_states=np.array(self._batches, dtype=object),
                input_ids=np.array(self._token_ids, dtype=object),
            )
        self._batches.clear()
        self._token_ids.clear()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NamedDimModel(nn.Module):
    def __init__(self, config: NamedDimConfig):
        super().__init__()
        self.config = config
        self.named_dims = config.named_dims

        hf_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.context_len,
            n_embd=config.hidden_dim,
            n_layer=config.n_layers,
            n_head=config.n_heads,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
        )
        self.lm = GPT2LMHeadModel(hf_config)

        self._activation_logger: ActivationLogger | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """Returns a HuggingFace CausalLMOutput.

        hidden_states[-1] is the final-layer hidden state (shape: B, T, H).
        Dimensions [0:3] of that are designated R, G, B.
        """
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        if self._activation_logger is not None:
            self._activation_logger.log(out.hidden_states[-1], input_ids)
        return out

    # ------------------------------------------------------------------
    # Named-dimension introspection
    # ------------------------------------------------------------------

    def named_dim_values(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice the named dimensions out of a hidden state.

        Args:
          hidden: shape (..., hidden_dim)
        Returns:
          dict {name: tensor of shape (...)}
        """
        return {name: hidden[..., idx] for name, idx in self.named_dims.items()}

    # ------------------------------------------------------------------
    # Activation logging
    # ------------------------------------------------------------------

    @contextmanager
    def log_activations_to(self, path: str | Path):
        """Context manager: log activations for forward passes inside the block,
        then flush to disk on exit."""
        logger = ActivationLogger(path)
        self._activation_logger = logger
        try:
            yield logger
        finally:
            logger.flush()
            self._activation_logger = None

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
