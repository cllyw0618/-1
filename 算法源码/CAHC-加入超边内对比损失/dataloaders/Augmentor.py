from typing import Tuple

import torch


class Augmentor:
    """Feature-mask and hyperedge-drop augmentation for contrastive views."""

    def __init__(self, feat_mask_rate: float = 0.3, edge_perturb_rate: float = 0.2):
        self.feat_mask_rate = feat_mask_rate
        self.edge_perturb_rate = edge_perturb_rate

    def _drop_features(self, x: torch.Tensor, p: float) -> torch.Tensor:
        drop_mask = torch.rand_like(x) > p
        return x * drop_mask

    def _drop_incidence(self, hyperedge_index: torch.Tensor, p: float) -> torch.Tensor:
        mask = torch.rand(hyperedge_index.shape[1], device=hyperedge_index.device) >= p
        return hyperedge_index[:, mask]

    def augment(
        self, features: torch.Tensor, hyperedge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # View 1
        features_1 = self._drop_features(features, self.feat_mask_rate)
        hyperedge_index_1 = self._drop_incidence(hyperedge_index, self.edge_perturb_rate)

        # View 2 (slightly different strength)
        feat_mask_rate_2 = min(0.5, self.feat_mask_rate * 1.5)
        edge_perturb_rate_2 = min(0.3, self.edge_perturb_rate * 0.8)
        features_2 = self._drop_features(features, feat_mask_rate_2)
        hyperedge_index_2 = self._drop_incidence(hyperedge_index, edge_perturb_rate_2)

        return features_1, hyperedge_index_1, features_2, hyperedge_index_2
