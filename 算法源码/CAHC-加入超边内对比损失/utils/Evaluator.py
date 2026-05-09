from typing import Dict, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


class SimpleHypergraphEvaluator:
    """Evaluate ACC/NMI/ARI/F1 for clustering outputs."""

    def evaluate(
        self,
        predicted_labels: Union[torch.Tensor, np.ndarray],
        ground_truth: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, float]:
        y_pred = predicted_labels.detach().cpu().numpy() if isinstance(predicted_labels, torch.Tensor) else np.asarray(predicted_labels)
        y_true = ground_truth.detach().cpu().numpy() if isinstance(ground_truth, torch.Tensor) else np.asarray(ground_truth)

        return {
            "acc": self._calculate_acc(y_true, y_pred),
            "nmi": normalized_mutual_info_score(y_true, y_pred),
            "ari": adjusted_rand_score(y_true, y_pred),
            "f1": self._calculate_f1(y_true, y_pred),
        }

    @staticmethod
    def _hungarian_mapping(y_true: np.ndarray, y_pred: np.ndarray):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        d = int(max(y_pred.max(), y_true.max()) + 1)
        w = np.zeros((d, d), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return w, row_ind, col_ind

    def _calculate_acc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        w, row_ind, col_ind = self._hungarian_mapping(y_true, y_pred)
        return float(w[row_ind, col_ind].sum() / y_pred.size)

    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _, row_ind, col_ind = self._hungarian_mapping(y_true, y_pred)
        mapped_pred = np.zeros_like(y_pred)
        for i, cluster_id in enumerate(row_ind):
            mapped_pred[y_pred == cluster_id] = col_ind[i]
        return float(f1_score(y_true, mapped_pred, average="macro"))
