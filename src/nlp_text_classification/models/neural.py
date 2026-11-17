"""TextCNN and packed recurrent encoders used by Batch A."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RecurrentClassifier(nn.Module):
    def __init__(self, model: str, vocab_size: int, embedding_dim: int,
                 hidden_size: int, outputs: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bidirectional = model == "bilstm"
        recurrent = nn.RNN if model == "vanilla_rnn" else nn.LSTM
        self.encoder = recurrent(
            embedding_dim, hidden_size, batch_first=True, bidirectional=self.bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size * (2 if self.bidirectional else 1), outputs)

    def representation(self, tokens, lengths):
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.to(device="cpu"), batch_first=True, enforce_sorted=False
        )
        _, state = self.encoder(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        if self.bidirectional:
            return torch.cat((hidden[-2], hidden[-1]), dim=1)
        return hidden[-1]

    def forward(self, tokens, lengths):
        return self.output(self.dropout(self.representation(tokens, lengths)))


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, kernels: list[int],
                 filters: int, outputs: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, filters, k) for k in kernels)
        self.kernels = kernels
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(filters * len(kernels), outputs)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens).transpose(1, 2)
        features = []
        for kernel, conv in zip(self.kernels, self.convs, strict=True):
            values = conv(embedded).relu()
            positions = torch.arange(values.shape[-1], device=tokens.device)
            windows = (lengths.to(tokens.device) - kernel + 1).clamp(min=1)
            mask = positions[None, :] >= windows[:, None]
            features.append(values.masked_fill(mask[:, None, :], 0).amax(dim=2))
        return self.output(self.dropout(torch.cat(features, dim=1)))


def make_model(model: str, settings: dict, vocab_size: int, outputs: int) -> nn.Module:
    common = (vocab_size, settings["embedding_dim"])
    if model == "textcnn":
        cnn = settings["textcnn"]
        return TextCNN(*common, cnn["kernel_sizes"], cnn["filters_per_size"],
                       outputs, settings["dropout"])
    if model not in {"vanilla_rnn", "lstm", "bilstm"}:
        raise ValueError(f"Unknown neural model: {model}")
    return RecurrentClassifier(model, *common, settings["recurrent"]["hidden_size"],
                               outputs, settings["dropout"])
