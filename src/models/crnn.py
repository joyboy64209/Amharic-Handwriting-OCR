"""CRNN sequence recognizer with CTC loss for short Amharic text lines.

Architecture:
    CNN feature extractor (4 conv blocks) -> sequence of feature columns
    -> bidirectional LSTM -> linear projection to (num_classes + 1) logits
    -> CTC loss / greedy / beam-search decoding.

Input:  (B, 1, H, W) grayscale line image.
Output: (B, T, num_classes + 1) log-probabilities over the extended alphabet
        (the extra index 0 is the CTC blank).

This mirrors the CNN's convolutional front end so the two models share a
feature representation, then models the character sequence explicitly rather
than cutting connected handwriting into isolated glyphs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import ConvBlock


class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_height: int = 28,
                 hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.blank = 0
        self.output_dim = num_classes + 1

        self.block1 = ConvBlock(1, 32, 0.1)
        self.block2 = ConvBlock(32, 64, 0.15)
        self.block3 = ConvBlock(64, 128, 0.2)
        self.block4 = ConvBlock(128, 256, 0.25)

        self.feature_dim = None
        self.rnn = None
        self.fc = nn.Linear(hidden * 2, self.output_dim)

        self._infer_feature_dim(img_height)

    def _infer_feature_dim(self, img_height: int) -> None:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_height, img_height)
            x = self.block4(self.block3(self.block2(self.block1(dummy))))
            b, c, hh, ww = x.shape
            self.feature_dim = c * hh
        self.rnn = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        b, c, hh, ww = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, ww, c * hh)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=2)
        return x


def ctc_loss(log_probs: torch.Tensor, targets: torch.Tensor,
             input_lengths: torch.Tensor, target_lengths: torch.Tensor,
             blank: int = 0) -> torch.Tensor:
    """Wrapper around nn.CTCLoss for clarity."""
    criterion = nn.CTCLoss(blank=blank, zero_infinity=True)
    return criterion(log_probs, targets, input_lengths, target_lengths)


def greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """Greedy (best-path) CTC decoding: argmax per step, remove blanks & repeats."""
    log_probs = log_probs.detach().cpu()
    best = log_probs.argmax(dim=2)
    decoded: list[list[int]] = []
    for seq in best:
        chars: list[int] = []
        prev = blank
        for idx in seq.tolist():
            if idx != blank and idx != prev:
                chars.append(idx)
            prev = idx
        decoded.append(chars)
    return decoded


def beam_search_decode(log_probs: torch.Tensor, beam_width: int = 10,
                       blank: int = 0) -> list[list[int]]:
    """Prefix-beam CTC decode (no language model). Returns top path per batch."""
    log_probs = log_probs.detach().cpu()
    batch = log_probs.shape[0]
    decoded: list[list[int]] = []
    for b in range(batch):
        seq = log_probs[b]
        beams: list[tuple[tuple[int, ...], float]] = [(( ), 0.0)]
        for t in range(seq.shape[0]):
            new_beams: dict[tuple[int, ...], float] = {}
            logp = seq[t]
            for prefix, score in beams:
                for c in range(logp.shape[0]):
                    ns = score + float(logp[c])
                    if c == blank:
                        key = prefix
                    elif prefix and prefix[-1] == c:
                        key = prefix
                    else:
                        key = prefix + (c,)
                    if key not in new_beams or ns > new_beams[key]:
                        new_beams[key] = ns
            beams = sorted(new_beams.items(), key=lambda kv: kv[1], reverse=True)[:beam_width]
        decoded.append(list(beams[0][0]) if beams else [])
    return decoded


__all__ = ["CRNN", "ctc_loss", "greedy_decode", "beam_search_decode"]