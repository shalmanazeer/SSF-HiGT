"""Evaluation metrics: accuracy, precision/recall/F1, false-positive rate,
and per-class breakdowns, matching the reporting style of Tables 4-7."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, benign_label: int = 0) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    # FPR over the benign class: FP / (FP + TN), treating "benign" as negative class
    benign_idx = benign_label
    fp = cm[:, benign_idx].sum() - cm[benign_idx, benign_idx]
    tn = cm.sum() - cm[benign_idx, :].sum() - cm[:, benign_idx].sum() + cm[benign_idx, benign_idx]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "fpr": fpr * 100,
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    report = {}
    for i, name in enumerate(class_names):
        report[name] = {
            "precision": precision[i] * 100,
            "recall": recall[i] * 100,
            "f1": f1[i] * 100,
            "support": int(support[i]),
        }
    return report
