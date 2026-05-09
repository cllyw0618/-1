# Project Context Summary 
--- 
 
## File: `submit_main.py` 
 
```python  
import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.cluster import KMeans
from torch import optim
import torch_scatter
from tqdm.auto import tqdm

from dataloaders import Augmentor, DatasetLoader
from model import EnhancedHGNN
from model.layers import ClusteringPrototypes, Loss
from utils import SimpleHypergraphEvaluator


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().item())
    return None


def format_metrics(metrics: dict, keys: Optional[list] = None) -> str:
    """Format numeric metric dict to compact printable text."""
    if keys is None:
        keys = list(metrics.keys())
    parts = []
    for key in keys:
        if key not in metrics:
            continue
        val = _to_float(metrics[key])
        if val is not None:
            parts.append(f"{key}={val:.4f}")
    return ", ".join(parts)


def fix_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_negative_samples(hyperedge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Create negative hyperedges by partially replacing nodes in each hyperedge."""
    node_idx, edge_idx = hyperedge_index
    num_edges = int(edge_idx.max().item()) + 1

    ones = torch.ones_like(edge_idx, dtype=torch.long)
    edge_sizes = torch_scatter.scatter_add(ones, edge_idx, dim=0, dim_size=num_edges)
    num_to_replace_per_edge = torch.log2(edge_sizes.float()).floor().long()
    num_to_replace_per_edge = torch.max(num_to_replace_per_edge, (edge_sizes > 1).long())

    indices_to_replace_list = []
    for edge_id in range(num_edges):
        num_replace = int(num_to_replace_per_edge[edge_id].item())
        if num_replace <= 0:
            continue
        current_indices = torch.where(edge_idx == edge_id)[0]
        sampled = random.sample(current_indices.tolist(), num_replace)
        indices_to_replace_list.extend(sampled)

    if not indices_to_replace_list:
        return hyperedge_index.clone()

    indices_to_replace = torch.tensor(indices_to_replace_list, device=node_idx.device, dtype=torch.long)
    replacement_nodes = torch.randint(0, num_nodes, (indices_to_replace.numel(),), device=node_idx.device)

    negative_hyperedge_index = hyperedge_index.clone()
    negative_hyperedge_index[0, indices_to_replace] = replacement_nodes
    return negative_hyperedge_index


def train_step(
    encoder: nn.Module,
    project_head: nn.Module,
    cluster_module: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: Loss,
    features: torch.Tensor,
    pos_hyperedge_index: torch.Tensor,
    neg_hyperedge_index: torch.Tensor,
    features_aug1: torch.Tensor,
    hyperedge_index_aug1: torch.Tensor,
    features_aug2: torch.Tensor,
    hyperedge_index_aug2: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    tau: float,
    use_attr_guidance: bool,
    lnahc_weight: float,
    ortho_weight: float,
    detach_affinity: bool,
    balance_loss_type: str,
    empty_cluster_rho: float,
) -> dict:
    """Run one optimization step for CAHC."""
    encoder.train()
    project_head.train()
    cluster_module.train()
    loss_fn.train()

    optimizer.zero_grad()

    pos_node_emb, _ = encoder(features, pos_hyperedge_index)
    aug_emb_1, _ = encoder(features_aug1, hyperedge_index_aug1)
    aug_emb_2, _ = encoder(features_aug2, hyperedge_index_aug2)

    proj_emb_1 = project_head(aug_emb_1)
    proj_emb_2 = project_head(aug_emb_2)
    clustering_logits = cluster_module(pos_node_emb)

    original_features = features if use_attr_guidance else None

    loss, metrics = loss_fn(
        pos_node_embeds=pos_node_emb,
        pos_hyperedge_embeds=proj_emb_1,
        neg_hyperedge_embeds=proj_emb_2,
        pos_hyperedge_index=pos_hyperedge_index,
        neg_hyperedge_index=neg_hyperedge_index,
        clustering_logits=clustering_logits,
        original_features=original_features,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        tau=tau,
        lnahc_weight=lnahc_weight,
        ortho_weight=ortho_weight,
        detach_affinity=detach_affinity,
        balance_loss_type=balance_loss_type,
        empty_cluster_rho=empty_cluster_rho,
    )

    loss.backward()
    optimizer.step()

    metrics["total_loss"] = float(loss.item())
    return metrics


def evaluate(
    encoder: nn.Module,
    cluster_module: nn.Module,
    features: torch.Tensor,
    hyperedge_index: torch.Tensor,
    labels: torch.Tensor,
    evaluator: SimpleHypergraphEvaluator,
) -> dict:
    """Evaluate clustering metrics from cluster module predictions."""
    encoder.eval()
    cluster_module.eval()
    with torch.no_grad():
        node_emb, _ = encoder(features, hyperedge_index)
        logits = cluster_module(node_emb)
        preds = torch.argmax(logits, dim=1)
    return evaluator.evaluate(preds, labels)


def run_experiment(config: dict) -> dict:
    """Main training chain: data -> augment -> encode -> pretrain -> KMeans init -> finetune -> evaluate."""
    show_progress = bool(config.get("show_progress", True))
    log_interval = max(1, int(config.get("log_interval", 10)))
    print_config = bool(config.get("print_config", True))

    if print_config:
        print("========== Experiment Config ==========")
        summary_keys = [
            "dataset_name",
            "device",
            "seed",
            "emb_dim",
            "num_layers",
            "pretrain_epochs",
            "finetune_epochs",
            "lr",
            "weight_decay",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "tau",
            "lnahc_weight",
            "feat_mask_rate",
            "edge_drop_rate",
        ]
        for k in summary_keys:
            print(f"{k}: {config[k]}")
        print("======================================")

    fix_seed(int(config["seed"]))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(config['device'])}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    data = DatasetLoader().load(config["dataset_name"]).to(device)
    features = data.features
    hyperedge_index = data.hyperedge_index
    labels = data.labels

    num_nodes = int(features.shape[0])
    node_dim = int(features.shape[1])
    n_clusters = int(torch.unique(labels).numel())
    num_hyperedges = int(hyperedge_index[1].max().item()) + 1

    print(
        f"[Data] dataset={config['dataset_name']} | device={device} | "
        f"nodes={num_nodes} | hyperedges={num_hyperedges} | feat_dim={node_dim} | clusters={n_clusters}"
    )

    augmentor = Augmentor(config["feat_mask_rate"], config["edge_drop_rate"])
    encoder = EnhancedHGNN(node_dim, config["emb_dim"], config["num_layers"]).to(device)
    cluster_module = ClusteringPrototypes(n_clusters, config["emb_dim"], config["tau"]).to(device)

    project_head = nn.Sequential(
        nn.Linear(config["emb_dim"], config["emb_dim"]),
        nn.ReLU(),
        nn.Linear(config["emb_dim"], config["emb_dim"]),
    ).to(device)

    loss_fn = Loss(config["emb_dim"], config["emb_dim"]).to(device)
    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(project_head.parameters())
        + list(cluster_module.parameters())
        + list(loss_fn.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)
    evaluator = SimpleHypergraphEvaluator()

    # Stage 1: pretraining (no clustering loss, no LNAHC loss)
    print(f"[Stage 1/2] Pretraining for {config['pretrain_epochs']} epochs ...")
    pretrain_pbar = tqdm(
        range(config["pretrain_epochs"]),
        desc="Pretrain",
        disable=not show_progress,
    )
    for epoch in pretrain_pbar:
        neg_hyperedge_index = create_negative_samples(hyperedge_index, num_nodes)
        feat_aug1, edge_aug1, feat_aug2, edge_aug2 = augmentor.augment(features, hyperedge_index)

        pretrain_metrics = train_step(
            encoder=encoder,
            project_head=project_head,
            cluster_module=cluster_module,
            optimizer=optimizer,
            loss_fn=loss_fn,
            features=features,
            pos_hyperedge_index=hyperedge_index,
            neg_hyperedge_index=neg_hyperedge_index,
            features_aug1=feat_aug1,
            hyperedge_index_aug1=edge_aug1,
            features_aug2=feat_aug2,
            hyperedge_index_aug2=edge_aug2,
            alpha=config["alpha"],
            beta=config["beta"],
            gamma=0.0,
            delta=config["delta"],
            tau=config["tau"],
            use_attr_guidance=config["use_attr_guidance"],
            lnahc_weight=0.0,
            ortho_weight=config["ortho_weight"],
            detach_affinity=config["detach_affinity"],
            balance_loss_type=config["balance_loss_type"],
            empty_cluster_rho=config["empty_cluster_rho"],
        )
        if show_progress:
            pretrain_loss = _to_float(pretrain_metrics.get("total_loss", 0.0))
            if pretrain_loss is None:
                pretrain_loss = 0.0
            pretrain_pbar.set_postfix(
                loss=f"{pretrain_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        if (epoch + 1) % log_interval == 0 or epoch == 0 or (epoch + 1) == config["pretrain_epochs"]:
            text = format_metrics(pretrain_metrics)
            print(f"[Pretrain][{epoch + 1}/{config['pretrain_epochs']}] {text}")

    # KMeans initialization for cluster centers
    print("[Init] Running KMeans to initialize cluster prototypes ...")
    encoder.eval()
    with torch.no_grad():
        initial_node_emb, _ = encoder(features, hyperedge_index)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=int(config["seed"]))
    kmeans.fit(initial_node_emb.detach().cpu().numpy())
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    cluster_module.prototypes.data.copy_(centers)

    # Stage 2: fine-tuning
    print(f"[Stage 2/2] Fine-tuning for {config['finetune_epochs']} epochs ...")
    finetune_pbar = tqdm(
        range(config["finetune_epochs"]),
        desc="Finetune",
        disable=not show_progress,
    )
    for epoch in finetune_pbar:
        neg_hyperedge_index = create_negative_samples(hyperedge_index, num_nodes)
        feat_aug1, edge_aug1, feat_aug2, edge_aug2 = augmentor.augment(features, hyperedge_index)

        metrics = train_step(
            encoder=encoder,
            project_head=project_head,
            cluster_module=cluster_module,
            optimizer=optimizer,
            loss_fn=loss_fn,
            features=features,
            pos_hyperedge_index=hyperedge_index,
            neg_hyperedge_index=neg_hyperedge_index,
            features_aug1=feat_aug1,
            hyperedge_index_aug1=edge_aug1,
            features_aug2=feat_aug2,
            hyperedge_index_aug2=edge_aug2,
            alpha=config["alpha"],
            beta=config["beta"],
            gamma=config["gamma"],
            delta=config["delta"],
            tau=config["tau"],
            use_attr_guidance=config["use_attr_guidance"],
            lnahc_weight=config["lnahc_weight"],
            ortho_weight=config["ortho_weight"],
            detach_affinity=config["detach_affinity"],
            balance_loss_type=config["balance_loss_type"],
            empty_cluster_rho=config["empty_cluster_rho"],
        )
        scheduler.step(metrics["total_loss"])
        if show_progress:
            finetune_loss = _to_float(metrics.get("total_loss", 0.0))
            if finetune_loss is None:
                finetune_loss = 0.0
            finetune_pbar.set_postfix(
                loss=f"{finetune_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        if (epoch + 1) % log_interval == 0 or epoch == 0 or (epoch + 1) == config["finetune_epochs"]:
            text = format_metrics(metrics)
            print(f"[Finetune][{epoch + 1}/{config['finetune_epochs']}] {text}")

    print("[Eval] Calculating final clustering metrics ...")
    final_metrics = evaluate(
        encoder=encoder,
        cluster_module=cluster_module,
        features=features,
        hyperedge_index=hyperedge_index,
        labels=labels,
        evaluator=evaluator,
    )
    return final_metrics


def load_config(config_path: str) -> dict:
    """Load YAML config and keep only required submission keys."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required_keys = [
        "dataset_name",
        "device",
        "seed",
        "emb_dim",
        "num_layers",
        "pretrain_epochs",
        "finetune_epochs",
        "lr",
        "weight_decay",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "tau",
        "lnahc_weight",
        "ortho_weight",
        "detach_affinity",
        "balance_loss_type",
        "empty_cluster_rho",
        "use_attr_guidance",
        "feat_mask_rate",
        "edge_drop_rate",
    ]
    optional_defaults = {
        "show_progress": True,
        "log_interval": 10,
        "print_config": True,
    }

    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    for key, default_val in optional_defaults.items():
        config.setdefault(key, default_val)

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal CAHC submission training entry")
    parser.add_argument("--config", type=str, default="submit_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        script_relative_config = Path(__file__).resolve().parent / config_path
        if script_relative_config.exists():
            config_path = script_relative_config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(str(config_path))
    metrics = run_experiment(config)

    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"ACC: {metrics['acc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
 
``` 
 
## File: `submit_config.yaml` 
 
```yaml  
﻿# 数据集名称（如 cora / citeseer / pubmed）
dataset_name: cora_coauthor

# 设备编号：0 表示 cuda:0；无 GPU 时自动回退到 CPU
device: 0

# 随机种子（保证可复现）
seed: 42

# 编码器隐藏维度
emb_dim: 128

# 编码层数（推荐 1，避免过平滑）
num_layers: 1

# 预训练轮数（阶段1）
pretrain_epochs: 260

# 微调轮数（阶段2）
finetune_epochs: 100

# 学习率
lr: 0.000312

# L2 正则
weight_decay: 5.0e-05

# 结构判别损失权重
alpha: 1.0

# InfoNCE 对比损失权重
beta: 6.8354

# 聚类自监督损失权重
gamma: 0.0

# 超边内对比（intra）损失权重
delta: 0.0811

# 对比学习温度系数
tau: 0.4151

# LNAHC 总权重（0 表示关闭）
lnahc_weight: 1.0807

# LNAHC 平衡项权重
ortho_weight: 0.85

# 动态亲和图是否 detach（推荐 true）
detach_affinity: true

# 平衡损失类型：ortho 或 empty
balance_loss_type: ortho

# empty 平衡项阈值（仅 balance_loss_type=empty 时生效）
empty_cluster_rho: 0.02

# 是否启用属性引导（intra 中使用原始特征相似度）
use_attr_guidance: true

# 特征 mask 比例（数据增强）
feat_mask_rate: 0.4

# 超边 drop 比例（数据增强）
edge_drop_rate: 0.2

# 是否显示 tqdm 进度条
show_progress: true

# 日志打印间隔（每 N 个 epoch 打印一次）
log_interval: 10

# 是否在启动时打印核心参数摘要
print_config: true
# --- bayesian 结果参数备选（截图中的另外两组）---
# 方案2：
# lr: 0.000292
# beta: 5.1767
# delta: 0.0741
# tau: 0.4779
# lnahc_weight: 0.5288
#
# 方案3：
# lr: 0.000343
# beta: 3.7163
# delta: 0.1165
# tau: 0.6107
# lnahc_weight: 0.8288
 
``` 
 
## File: `data\graph_dataset.py` 
 
```python  
﻿from pathlib import Path
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
 
``` 
 
## File: `data\__init__.py` 
 
```python  
﻿# Data package for hypergraph dataset loading.
 
``` 
 
## File: `dataloaders\Augmentor.py` 
 
```python  
﻿from typing import Tuple

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
 
``` 
 
## File: `dataloaders\data_loader.py` 
 
```python  
﻿from data.graph_dataset import GraphDataset


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
 
``` 
 
## File: `dataloaders\__init__.py` 
 
```python  
from .data_loader import DatasetLoader
from .Augmentor import Augmentor

__all__ = ["DatasetLoader", "Augmentor"]
 
``` 
 
## File: `model\EnhancedHGNN.py` 
 
```python  
﻿import torch
import torch.nn as nn
from torch import Tensor

from model.EnhancedHGNNConv import EnhancedHGNNConv


class EnhancedHGNN(nn.Module):
    """Enhanced HGNN encoder used by the final method."""

    def __init__(
        self,
        node_dim: int,
        emb_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, emb_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                EnhancedHGNNConv(
                    node_in_dim=emb_dim,
                    node_out_dim=emb_dim,
                    hyper_in_dim=emb_dim,
                    hyper_out_dim=emb_dim,
                    drop_rate=0.1,
                    is_last=(i == num_layers - 1),
                    num_heads=num_heads,
                    attn_dim=attn_dim,
                )
            )

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> tuple[Tensor, Tensor]:
        x = self.node_encoder(x)
        y = self._init_hyper_emb(x, hyperedge_index)
        for layer in self.layers:
            x, y = layer(x, y, hyperedge_index)
        return x, y

    @staticmethod
    def _init_hyper_emb(x: Tensor, hyperedge_index: Tensor) -> Tensor:
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = int(edge_idx.max().item()) + 1

        counts = torch.zeros(num_hyperedges, device=x.device)
        counts.scatter_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.float))
        counts = torch.clamp(counts, min=1)

        y = torch.zeros(num_hyperedges, x.size(1), device=x.device)
        y.scatter_add_(0, edge_idx.unsqueeze(1).expand(-1, x.size(1)), x[node_idx])
        y = y / counts.unsqueeze(1)
        return y
 
``` 
 
## File: `model\EnhancedHGNNConv.py` 
 
```python  
﻿import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor, matmul


class EnhancedHGNNConv(nn.Module):
    """Bidirectional node-hyperedge attention convolution."""

    def __init__(
        self,
        node_in_dim: int,
        node_out_dim: int,
        hyper_in_dim: int,
        hyper_out_dim: int,
        bias: bool = True,
        drop_rate: float = 0.4,
        is_last: bool = False,
        num_heads: int = 4,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.is_last = is_last
        self.num_heads = num_heads

        self.node_v_lin = nn.Linear(node_in_dim, node_out_dim, bias=bias)
        self.hyper_v_lin = nn.Linear(hyper_in_dim, hyper_out_dim, bias=bias)

        self.attn_dim = attn_dim
        self.head_dim = self.attn_dim // num_heads

        self.node_attn_proj = nn.Linear(node_out_dim, self.attn_dim, bias=False)
        self.hyper_attn_proj = nn.Linear(hyper_out_dim, self.attn_dim, bias=False)

        self.attn_vec_e2n = Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        self.attn_vec_n2e = Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.node_norm = nn.LayerNorm(node_out_dim)
        self.hyper_norm = nn.LayerNorm(hyper_out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_v_lin.weight)
        nn.init.xavier_uniform_(self.hyper_v_lin.weight)
        nn.init.xavier_uniform_(self.node_attn_proj.weight)
        nn.init.xavier_uniform_(self.hyper_attn_proj.weight)
        nn.init.xavier_uniform_(self.attn_vec_e2n)
        nn.init.xavier_uniform_(self.attn_vec_n2e)

    def forward(self, x: torch.Tensor, y: torch.Tensor, hyperedge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes, num_edges = x.size(0), y.size(0)

        x_trans = self.node_v_lin(x)
        y_trans = self.hyper_v_lin(y)

        x_proj = self.node_attn_proj(x_trans).view(num_nodes, self.num_heads, self.head_dim)
        y_proj = self.hyper_attn_proj(y_trans).view(num_edges, self.num_heads, self.head_dim)

        inc_t = SparseTensor(row=hyperedge_index[1], col=hyperedge_index[0], sparse_sizes=(num_edges, num_nodes))
        adj_t = inc_t.t()

        # E -> N
        node_queries = x_proj[adj_t.storage.row()]
        hyper_keys = y_proj[adj_t.storage.col()]
        attn_input_e2n = torch.cat([node_queries, hyper_keys], dim=-1)
        scores_e2n = (attn_input_e2n * self.attn_vec_e2n).sum(dim=-1)
        scores_e2n = F.leaky_relu(scores_e2n, 0.2)

        x_agg_list = []
        for head in range(self.num_heads):
            attn_weights = softmax(scores_e2n[:, head] / self.temperature, adj_t.storage.row(), num_nodes=num_nodes)
            attn_adj_t = adj_t.set_value(attn_weights, layout="coo")
            x_agg_list.append(matmul(attn_adj_t, y_trans, reduce="sum"))
        x_agg = torch.mean(torch.stack(x_agg_list, dim=0), dim=0)

        # N -> E
        hyper_queries = y_proj[inc_t.storage.row()]
        node_keys = x_proj[inc_t.storage.col()]
        attn_input_n2e = torch.cat([hyper_queries, node_keys], dim=-1)
        scores_n2e = (attn_input_n2e * self.attn_vec_n2e).sum(dim=-1)
        scores_n2e = F.leaky_relu(scores_n2e, 0.2)

        y_agg_list = []
        for head in range(self.num_heads):
            attn_weights = softmax(scores_n2e[:, head] / self.temperature, inc_t.storage.row(), num_nodes=num_edges)
            attn_inc_t = inc_t.set_value(attn_weights, layout="coo")
            y_agg_list.append(matmul(attn_inc_t, x_trans, reduce="sum"))
        y_agg = torch.mean(torch.stack(y_agg_list, dim=0), dim=0)

        x_final = self.node_norm(x_trans + x_agg)
        y_final = self.hyper_norm(y_trans + y_agg)

        if not self.is_last:
            x_final = self.drop(self.act(x_final))
            y_final = self.drop(self.act(y_final))

        return x_final, y_final
 
``` 
 
## File: `model\__init__.py` 
 
```python  
﻿from .EnhancedHGNN import EnhancedHGNN

__all__ = ["EnhancedHGNN"]
 
``` 
 
## File: `model\layers\ClusteringLayer.py` 
 
```python  
﻿import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusteringPrototypes(nn.Module):
    """Learnable cluster prototypes with cosine similarity logits."""

    def __init__(self, n_clusters: int, emb_dim: int, temperature: float = 0.5):
        super().__init__()
        self.prototypes = nn.Parameter(torch.Tensor(n_clusters, emb_dim))
        self.temperature = temperature
        nn.init.xavier_uniform_(self.prototypes.data)

    def forward(self, node_embeds: torch.Tensor) -> torch.Tensor:
        node_embeds_norm = F.normalize(node_embeds, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        similarity_matrix = torch.matmul(node_embeds_norm, prototypes_norm.t())
        return similarity_matrix / self.temperature
 
``` 
 
## File: `model\layers\Loss.py` 
 
```python  
﻿import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean


class AdvancedScorer(nn.Module):
    """Discriminator that scores whether a hyperedge is real or negative."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeds: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = int(edge_idx.max().item()) + 1

        mean_pooled = scatter_mean(node_embeds[node_idx], edge_idx, dim=0, dim_size=num_hyperedges)
        max_pooled = scatter_max(node_embeds[node_idx], edge_idx, dim=0, dim_size=num_hyperedges)[0]
        hyperedge_representation = torch.cat([mean_pooled, max_pooled], dim=1)
        return self.mlp(hyperedge_representation).squeeze(-1)


class Loss(nn.Module):
    """Total loss module for CAHC training."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.scorer = AdvancedScorer(embed_dim, hidden_dim)
        self.clustering_loss_fn = nn.CrossEntropyLoss()

    def compute_latent_lnahc_cut_loss(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        threshold: float = 0.5,
        detach_affinity: bool = True,
    ) -> torch.Tensor:
        device = z.device
        num_clusters = s.shape[1]

        z_for_graph = z.detach() if detach_affinity else z
        z_norm = F.normalize(z_for_graph, p=2, dim=1)
        a_dynamic = torch.mm(z_norm, z_norm.t())
        a_dynamic = torch.where(a_dynamic > threshold, a_dynamic, torch.zeros_like(a_dynamic))
        a_dynamic.fill_diagonal_(0.0)

        d_pi = a_dynamic.sum(dim=1)
        d_pi_stable = d_pi + 1e-4 * d_pi.mean()

        ds = s * d_pi_stable.unsqueeze(1)
        vol_mat = torch.mm(s.t(), ds)

        as_mat = torch.mm(a_dynamic, s)
        stas_mat = torch.mm(s.t(), as_mat)
        cut_mat = vol_mat - stas_mat

        identity = torch.eye(num_clusters, device=device)
        return torch.trace(torch.linalg.pinv(vol_mat + 1e-4 * identity) @ cut_mat)

    @staticmethod
    def compute_latent_ortho_loss(s: torch.Tensor) -> torch.Tensor:
        n_nodes = s.shape[0]
        n_clusters = s.shape[1]
        c_t_c = torch.matmul(s.t(), s) / n_nodes
        target = torch.eye(n_clusters, device=s.device) / n_clusters
        return torch.norm(c_t_c - target, p="fro") ** 2

    @staticmethod
    def compute_empty_cluster_loss(s: torch.Tensor, rho: float = 0.02) -> torch.Tensor:
        cluster_mass = s.mean(dim=0)
        return torch.relu(rho - cluster_mass).pow(2).mean()

    @staticmethod
    def compute_infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        sim_matrix = torch.matmul(z1, z2.t()) / temperature
        labels = torch.arange(sim_matrix.shape[0], device=z1.device)
        return (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2

    @staticmethod
    def compute_intra_view_loss(
        embeds: torch.Tensor,
        hyperedge_index: torch.Tensor,
        original_features: torch.Tensor = None,
        tau: float = 0.5,
    ) -> torch.Tensor:
        z = F.normalize(embeds, p=2, dim=1)
        sim_matrix = torch.matmul(z, z.t()) / tau

        num_nodes = z.shape[0]
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = int(edge_idx.max().item()) + 1

        values = torch.ones_like(node_idx, dtype=torch.float32)
        h_sparse = torch.sparse_coo_tensor(
            hyperedge_index,
            values,
            size=(num_nodes, num_hyperedges),
            device=embeds.device,
        )

        a = torch.sparse.mm(h_sparse, h_sparse.t().to_dense())
        mask = a > 0

        if original_features is not None:
            x_norm = F.normalize(original_features, p=2, dim=1)
            attr_sim = torch.matmul(x_norm, x_norm.t())
            repulsion_weight = (1.0 - attr_sim) / 2.0

            exp_sim = torch.exp(sim_matrix) * repulsion_weight
            exp_sim.fill_diagonal_(0)
            diag_exp = torch.exp(torch.diag(sim_matrix))
            exp_sim = exp_sim + torch.diag(diag_exp)
        else:
            exp_sim = torch.exp(sim_matrix)

        mask.fill_diagonal_(True)
        denom = (exp_sim * mask).sum(dim=1)
        num = torch.exp(torch.diag(sim_matrix))
        return -torch.log(num / (denom + 1e-8)).mean()

    def forward(
        self,
        pos_node_embeds: torch.Tensor,
        pos_hyperedge_embeds: torch.Tensor,
        neg_hyperedge_embeds: torch.Tensor,
        pos_hyperedge_index: torch.Tensor,
        neg_hyperedge_index: torch.Tensor,
        clustering_logits: torch.Tensor,
        original_features: torch.Tensor = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 0.0001,
        tau: float = 0.8,
        lnahc_weight: float = 0.0,
        ortho_weight: float = 1.0,
        detach_affinity: bool = True,
        balance_loss_type: str = "ortho",
        empty_cluster_rho: float = 0.02,
    ) -> tuple[torch.Tensor, dict]:
        total_loss = 0.0
        metrics = {}

        # A. Structure discrimination loss
        if alpha > 0:
            pos_scores = self.scorer(pos_node_embeds, pos_hyperedge_index)
            neg_scores = self.scorer(pos_node_embeds, neg_hyperedge_index)
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            structure_loss = (pos_loss + neg_loss) / 2.0

            total_loss += alpha * structure_loss
            metrics["structure_loss"] = float(structure_loss.item())
            metrics["pos_score"] = float(torch.sigmoid(pos_scores).mean().item())
            metrics["neg_score"] = float(torch.sigmoid(neg_scores).mean().item())
        else:
            metrics["structure_loss"] = 0.0

        # B. Inter-view contrastive loss (InfoNCE)
        if beta > 0:
            infonce_loss = self.compute_infonce_loss(pos_hyperedge_embeds, neg_hyperedge_embeds, tau)
            total_loss += beta * infonce_loss
            metrics["infonce_loss"] = float(infonce_loss.item())
        else:
            metrics["infonce_loss"] = 0.0

        # C. Clustering self-training loss
        if gamma > 0:
            with torch.no_grad():
                pseudo_labels = torch.argmax(clustering_logits, dim=1)
            clustering_loss = self.clustering_loss_fn(clustering_logits, pseudo_labels)
            total_loss += gamma * clustering_loss
            metrics["clustering_loss"] = float(clustering_loss.item())
        else:
            metrics["clustering_loss"] = 0.0

        # D. Intra-hyperedge contrastive loss
        if delta > 0:
            intra_loss = self.compute_intra_view_loss(
                embeds=pos_node_embeds,
                hyperedge_index=pos_hyperedge_index,
                original_features=original_features,
                tau=tau,
            )
            total_loss += delta * intra_loss
            metrics["intra_loss"] = float(intra_loss.item())
        else:
            metrics["intra_loss"] = 0.0

        # E. LNAHC latent cut + balance loss
        if lnahc_weight > 0:
            s = torch.softmax(clustering_logits, dim=-1)
            loss_cut = self.compute_latent_lnahc_cut_loss(
                pos_node_embeds,
                s,
                threshold=0.5,
                detach_affinity=detach_affinity,
            )

            if balance_loss_type == "empty":
                loss_balance = self.compute_empty_cluster_loss(s, rho=empty_cluster_rho)
            elif balance_loss_type == "ortho":
                loss_balance = self.compute_latent_ortho_loss(s)
            else:
                raise ValueError(f"Unsupported balance_loss_type: {balance_loss_type}")

            total_lnahc = loss_cut + ortho_weight * loss_balance
            total_loss += lnahc_weight * total_lnahc

            metrics["lnahc_cut_loss"] = float(loss_cut.item())
            metrics["lnahc_balance_loss"] = float(loss_balance.item())
            metrics["lnahc_total_loss"] = float(total_lnahc.item())
            metrics["balance_loss_type"] = balance_loss_type
            metrics["lnahc_ortho_loss"] = float(loss_balance.item()) if balance_loss_type != "empty" else 0.0
            metrics["lnahc_empty_loss"] = float(loss_balance.item()) if balance_loss_type == "empty" else 0.0
        else:
            metrics["lnahc_cut_loss"] = 0.0
            metrics["lnahc_balance_loss"] = 0.0
            metrics["lnahc_total_loss"] = 0.0
            metrics["balance_loss_type"] = balance_loss_type
            metrics["lnahc_ortho_loss"] = 0.0
            metrics["lnahc_empty_loss"] = 0.0

        if not isinstance(total_loss, torch.Tensor):
            total_loss = pos_node_embeds.sum() * 0.0

        return total_loss, metrics
 
``` 
 
## File: `model\layers\__init__.py` 
 
```python  
from .Loss import Loss
from .ClusteringLayer import ClusteringPrototypes

__all__ = ["Loss", "ClusteringPrototypes"]
 
``` 
 
## File: `utils\Evaluator.py` 
 
```python  
﻿from typing import Dict, Union

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
 
``` 
 
## File: `utils\__init__.py` 
 
```python  
﻿from .Evaluator import SimpleHypergraphEvaluator

__all__ = ["SimpleHypergraphEvaluator"]
 
``` 
 
