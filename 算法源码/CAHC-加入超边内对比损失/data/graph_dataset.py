from pathlib import Path
import pickle

import torch
from torch_scatter import scatter_add


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GraphDataset:
    """Minimal hypergraph dataset object used by submit pipeline."""

    def __init__(self, dataset_type: str, dataset_name: str, device: str = "cpu"):
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.device = device

        if dataset_type in {"cocitation", "coauthorship"}:
            self.dataset_dir = _PROJECT_ROOT / "data" / "dataset" / dataset_type / dataset_name
        else:
            self.dataset_dir = _PROJECT_ROOT / "data" / "dataset" / dataset_name

        self._load_raw()
        self._preprocess()
        self.to(device)

    def _load_raw(self) -> None:
        with open(self.dataset_dir / "features.pickle", "rb") as f:
            self.features = pickle.load(f)
        with open(self.dataset_dir / "hypergraph.pickle", "rb") as f:
            self.hypergraph = pickle.load(f)
        with open(self.dataset_dir / "labels.pickle", "rb") as f:
            self.labels = pickle.load(f)

    def _preprocess(self) -> None:
        edge_ids = {edge: idx for idx, edge in enumerate(self.hypergraph.keys())}
        incidence_pairs = []
        for edge, nodes in self.hypergraph.items():
            edge_idx = edge_ids[edge]
            for node in nodes:
                incidence_pairs.append([node, edge_idx])

        features = self.features.toarray() if hasattr(self.features, "toarray") else self.features
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.hyperedge_index = torch.as_tensor(incidence_pairs, dtype=torch.long).t().contiguous()
        self.labels = torch.as_tensor(self.labels, dtype=torch.long)

        self.num_nodes = int(self.hyperedge_index[0].max()) + 1
        self.num_edges = int(self.hyperedge_index[1].max()) + 1

        ones = torch.ones(self.hyperedge_index.shape[1], dtype=torch.float32)
        self.node_degree = scatter_add(ones, self.hyperedge_index[0], dim=0, dim_size=self.num_nodes)
        self.edge_size = scatter_add(ones, self.hyperedge_index[1], dim=0, dim_size=self.num_edges)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self
