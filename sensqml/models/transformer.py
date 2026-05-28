"""
Transformer encoder classifier.

Token embeddings + positional vectors → multi-head self-attention stack
→ CLS-style mean pool over unpadded positions → linear head.

Uses PyTorch's built-in `TransformerEncoder`; positional information comes from
the `Sequence.positions` array produced by SequenceBuilder (which may be
absolute sinusoidal, relative, or time-delta-aware).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sensqml.features.sequence_builder import Sequence
from sensqml.models.base import BaseSequenceModel
from sensqml.models.lstm import _stack_sequences  # reuse


class _TransformerNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        pos_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Project (embed + pos) into a uniform model dimension
        model_dim = embed_dim + pos_dim
        # nn.TransformerEncoder requires d_model % num_heads == 0
        assert model_dim % num_heads == 0, (
            f"embed_dim + pos_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, token_ids, positions, attention_mask):
        emb = self.embedding(token_ids)                       # (B, L, E)
        x = torch.cat([emb, positions], dim=-1)               # (B, L, E+P)
        # Transformer expects True for *masked* positions
        key_padding_mask = (attention_mask == 0)
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Masked mean pool
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(self.dropout(pooled))


class TransformerModel(BaseSequenceModel):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 48,
        pos_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.2,
        device: str | None = None,
    ):
        self.config = dict(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _TransformerNet(**self.config).to(self.device)

    def fit(
        self,
        sequences: list[Sequence],
        labels: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs,
    ) -> "TransformerModel":
        token_ids, positions, masks = _stack_sequences(sequences)
        y = torch.tensor(labels, dtype=torch.long)

        ds = TensorDataset(token_ids, positions, masks, y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        self.net.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_tokens, batch_pos, batch_mask, batch_y in loader:
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

            print(f"[Transformer] epoch {epoch+1}/{epochs}  train_loss={total_loss/len(ds):.4f}")

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

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "config": self.config}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TransformerModel":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"])
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.to(model.device)
        return model
