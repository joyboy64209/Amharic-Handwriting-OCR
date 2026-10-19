"""Evaluation metrics for Amharic OCR.

Classification (isolated characters):
    accuracy, precision, recall, F1 (macro + per-class), confusion matrix.

Sequence recognition (text lines):
    Character Error Rate (CER) and Word Error Rate (WER) via Levenshtein
    edit distance on normalized Amharic text.

All metrics are computed from real predictions/labels only. Nothing here is
fabricated; callers must pass actual evaluation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Classic Levenshtein distance between two sequences of tokens."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[m]


def normalize_text(text: str) -> str:
    """Normalize Amharic text for CER/WER: strip, collapse whitespace."""
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit distance / reference length."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(list(ref), list(hyp)) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = edit distance / reference word count."""
    ref = normalize_text(reference).split(" ") if normalize_text(reference) else []
    hyp = normalize_text(hypothesis).split(" ") if normalize_text(hypothesis) else []
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def cer_wer_list(references: Sequence[str], hypotheses: Sequence[str]
                 ) -> Tuple[float, float]:
    """Micro-averaged CER and WER across a list of (ref, hyp) pairs."""
    total_edit_c = 0
    total_len_c = 0
    total_edit_w = 0
    total_len_w = 0
    for ref, hyp in zip(references, hypotheses):
        ref_n = normalize_text(ref)
        hyp_n = normalize_text(hyp)
        total_edit_c += _levenshtein(list(ref_n), list(hyp_n))
        total_len_c += len(ref_n)
        ref_w = ref_n.split(" ") if ref_n else []
        hyp_w = hyp_n.split(" ") if hyp_n else []
        total_edit_w += _levenshtein(ref_w, hyp_w)
        total_len_w += len(ref_w)
    cer_v = total_edit_c / total_len_c if total_len_c else 0.0
    wer_v = total_edit_w / total_len_w if total_len_w else 0.0
    return cer_v, wer_v


@dataclass
class ClassificationReport:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_per_class: List[float] = field(default_factory=list)
    recall_per_class: List[float] = field(default_factory=list)
    f1_per_class: List[float] = field(default_factory=list)
    support_per_class: List[int] = field(default_factory=list)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)


def confusion_matrix_from_lists(y_true: Sequence[int], y_pred: Sequence[int],
                                num_classes: int) -> List[List[int]]:
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                           num_classes: int,
                           labels: Sequence[str] | None = None
                           ) -> ClassificationReport:
    cm = confusion_matrix_from_lists(y_true, y_pred, num_classes)
    cm_np = np.array(cm, dtype=np.float64)

    total = cm_np.sum()
    accuracy = float(np.trace(cm_np) / total) if total else 0.0

    precision_per_class: List[float] = []
    recall_per_class: List[float] = []
    f1_per_class: List[float] = []
    support_per_class: List[int] = []

    for i in range(num_classes):
        tp = cm_np[i, i]
        fn = cm_np[i, :].sum() - tp
        fp = cm_np[:, i].sum() - tp
        support = int(cm_np[i, :].sum())
        support_per_class.append(support)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) \
            if (precision + recall) > 0 else 0.0
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    def macro(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    return ClassificationReport(
        accuracy=accuracy,
        precision_macro=macro(precision_per_class),
        recall_macro=macro(recall_per_class),
        f1_macro=macro(f1_per_class),
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        f1_per_class=f1_per_class,
        support_per_class=support_per_class,
        confusion_matrix=cm,
        labels=list(labels) if labels is not None else [str(i) for i in range(num_classes)],
    )


__all__ = [
    "cer", "wer", "cer_wer_list", "normalize_text",
    "classification_metrics", "confusion_matrix_from_lists",
    "ClassificationReport",
]