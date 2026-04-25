from data.graph_dataset import GraphDataset


class DatasetLoader:
    """Factory loader for supported hypergraph datasets."""

    _DATASET_SPECS = {
        "cora": ("cocitation", "cora"),
        "citeseer": ("cocitation", "citeseer"),
        "pubmed": ("cocitation", "pubmed"),
        "cora_coauthor": ("coauthorship", "cora"),
        "dblp_coauthor": ("coauthorship", "dblp"),
        "zoo": ("etc", "zoo"),
        "20newsW100": ("etc", "20newsW100"),
        "Mushroom": ("etc", "Mushroom"),
        "NTU2012": ("cv", "NTU2012"),
        "ModelNet40": ("cv", "ModelNet40"),
        "yelpRestaurant": ("cv", "yelpRestaurant"),
    }

    def load(self, dataset_name: str = "cora") -> GraphDataset:
        if dataset_name not in self._DATASET_SPECS:
            supported = ", ".join(sorted(self._DATASET_SPECS.keys()))
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
        dataset_type, dataset_id = self._DATASET_SPECS[dataset_name]
        return GraphDataset(dataset_type=dataset_type, dataset_name=dataset_id)
