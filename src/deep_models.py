"""Compact PyTorch text classifiers: LSTM, multi-kernel CNN, and attention pooling."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMSentimentClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, num_classes: int, embedding_dim: int = 128,
        hidden_dim: int = 128, pad_idx: int = 0, dropout: float = 0.25,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        if lengths is not None:
            safe_lengths = lengths.clamp(min=1, max=input_ids.size(1)).cpu()
            packed = pack_padded_sequence(
                embedded, safe_lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.lstm.bidirectional else hidden[-1]
        return self.classifier(self.dropout(pooled))


class MultiKernelCNNClassifier(nn.Module):
    """Text CNN with parallel 3-, 4-, and 5-token filters by default."""
    def __init__(
        self, vocab_size: int, num_classes: int, embedding_dim: int = 128,
        num_filters: int = 96, kernel_sizes: tuple[int, ...] = (3, 4, 5),
        pad_idx: int = 0, dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if not kernel_sizes or any(k < 1 for k in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            x = embedded
            if x.size(-1) < conv.kernel_size[0]:
                x = nn.functional.pad(x, (0, conv.kernel_size[0] - x.size(-1)))
            pooled.append(torch.amax(torch.relu(conv(x)), dim=-1))
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class AttentionPoolingClassifier(nn.Module):
    """BiLSTM encoder with learned scalar attention over token states."""
    def __init__(
        self, vocab_size: int, num_classes: int, embedding_dim: int = 128,
        hidden_dim: int = 96, pad_idx: int = 0, dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        states, _ = self.encoder(self.embedding(input_ids))
        scores = self.attention(states).squeeze(-1)
        if lengths is not None:
            positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            mask = positions >= lengths.to(input_ids.device).unsqueeze(1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(states * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(context))
