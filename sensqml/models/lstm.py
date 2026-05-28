"""
LSTM sequence classifier.

Consumes raw Sequence objects, embeds token IDs, optionally concatenates
positional vectors, and runs a bidirectional LSTM with mean-pooling over the
unpadded positions before a linear head.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sensqml.features.sequence_builder import Sequence
from sensqml.models.base import BaseSequenceModel


class _LSTMNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        pos_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_dim = pos_dim
        lstm_input = embed_dim + pos_dim
        self.lstm = nn.LSTM(
            lstm_input,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, token_ids, positions, attention_mask):
        emb = self.embedding(token_ids)                # (B, L, E)
        x = torch.cat([emb, positions], dim=-1)        # (B, L, E+P)
        out, _ = self.lstm(x)                          # (B, L, H*)
        # Masked mean pool
        mask = attention_mask.unsqueeze(-1).float()    # (B, L, 1)
        summed = (out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        return self.head(self.dropout(pooled))


class LSTMModel(BaseSequenceModel):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        pos_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: str | None = None,
    ):
        self.config = dict(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            pos_dim=pos_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _LSTMNet(**self.config).to(self.device)

    # ------------------------------------------------------------------
    # Train / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: list[Sequence],
        labels: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_split: float = 0.1,
        **kwargs,
    ) -> "LSTMModel":
        token_ids, positions, masks = _stack_sequences(sequences)
        y = torch.tensor(labels, dtype=torch.long)

        ds = TensorDataset(token_ids, positions, masks, y)
        n_val = int(len(ds) * val_split)
        n_train = len(ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.net.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_tokens, batch_pos, batch_mask, batch_y in train_loader:
                batch_tokens = batch_tokens.to(self.device)
                batch_pos = batch_pos.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.net(batch_tokens, batch_pos, batch_mask)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * batch_y.size(0)

            avg_loss = total_loss / n_train
            print(f"[LSTM] epoch {epoch+1}/{epochs}  train_loss={avg_loss:.4f}")

        return self

    def predict_proba(self, sequences: list[Sequence]) -> np.ndarray:
        self.net.eval()
        token_ids, positions, masks = _stack_sequences(sequences)
        with torch.no_grad():
            logits = self.net(
                token_ids.to(self.device),
                positions.to(self.device),
                masks.to(self.device),
            )
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"])
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.to(model.device)
        return model


# ---------------------------------------------------------------------------
# Utility — stack list[Sequence] into batched tensors
# ---------------------------------------------------------------------------
def _stack_sequences(sequences: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = torch.tensor(np.stack([s.token_ids for s in sequences]), dtype=torch.long)
    positions = torch.tensor(np.stack([s.positions for s in sequences]), dtype=torch.float32)
    masks = torch.tensor(np.stack([s.attention_mask for s in sequences]), dtype=torch.float32)
    return token_ids, positions, masks
