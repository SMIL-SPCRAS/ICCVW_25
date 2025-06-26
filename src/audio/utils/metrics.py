import numpy as np
from sklearn.metrics import f1_score, recall_score
from typing import Any


class UAR:
    """Unweighted Average Recall (UAR) metric."""
    def __init__(self, task: str) -> None:
        self.task = task

    def calc(self, y_true: Any, y_pred: Any) -> float:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_pred_arr.ndim > 1:
            y_pred_arr = y_pred_arr.argmax(axis=1)
        if y_true_arr.ndim > 1:
            y_true_arr = y_true_arr.argmax(axis=1)

        return recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)

    def __str__(self) -> str:
        return f"uar_{self.task}"


class MacroF1:
    """Macro-averaged F1 score metric."""
    def __init__(self, task: str) -> None:
        self.task = task

    def calc(self, y_true: Any, y_pred: Any) -> float:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if y_pred_arr.ndim > 1:
            y_pred_arr = y_pred_arr.argmax(axis=1)
        if y_true_arr.ndim > 1:
            y_true_arr = y_true_arr.argmax(axis=1)

        return f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)

    def __str__(self) -> str:
        return f"macro_f1_{self.task}"
