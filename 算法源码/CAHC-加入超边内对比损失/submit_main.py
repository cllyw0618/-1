import argparse
import json
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
from export_demo_json import convert_to_demo_json
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


def resolve_dataset_pickle_paths(dataset_name: str) -> dict:
    specs = DatasetLoader._DATASET_SPECS
    if dataset_name not in specs:
        supported = ", ".join(sorted(specs.keys()))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")

    dataset_type, dataset_id = specs[dataset_name]
    project_root = Path(__file__).resolve().parent
    if dataset_type in {"cocitation", "coauthorship"}:
        dataset_dir = project_root / "data" / "dataset" / dataset_type / dataset_id
    else:
        dataset_dir = project_root / "data" / "dataset" / dataset_id

    return {
        "dataset_id": dataset_id,
        "dataset_dir": dataset_dir,
        "features": dataset_dir / "features.pickle",
        "hypergraph": dataset_dir / "hypergraph.pickle",
        "labels": dataset_dir / "labels.pickle",
    }


def to_display_dataset_name(dataset_id: str) -> str:
    if not dataset_id:
        return "Dataset"
    if dataset_id[0].islower():
        return dataset_id[0].upper() + dataset_id[1:]
    return dataset_id


def run_experiment(config: dict) -> dict:
    """Main training chain: data -> augment -> encode -> pretrain -> KMeans init -> finetune -> evaluate."""
    show_progress = bool(config.get("show_progress", True))
    log_interval = max(1, int(config.get("log_interval", 10)))
    print_config = bool(config.get("print_config", True))
    use_kmeans_init = bool(config.get("use_kmeans_init", True))

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
            "use_kmeans_init",
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
    encoder.eval()
    with torch.no_grad():
        initial_node_emb, _ = encoder(features, hyperedge_index)
    if use_kmeans_init:
        print("[Init] Running KMeans to initialize cluster prototypes ...")
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=int(config["seed"]))
        kmeans.fit(initial_node_emb.detach().cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        cluster_module.prototypes.data.copy_(centers)
    else:
        print("[Init] Skip KMeans prototype initialization; keep default learnable prototypes.")

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
    encoder.eval()
    cluster_module.eval()
    with torch.no_grad():
        final_node_emb, _ = encoder(features, hyperedge_index)
        final_logits = cluster_module(final_node_emb)
        final_preds = torch.argmax(final_logits, dim=1)
        final_probs = torch.softmax(final_logits, dim=1)

    final_metrics = evaluator.evaluate(final_preds, labels)
    return {
        "metrics": final_metrics,
        "final_emb": final_node_emb.detach().cpu().numpy(),
        "preds": final_preds.detach().cpu().numpy(),
        "probs": final_probs.detach().cpu().numpy(),
        "labels": labels.detach().cpu().numpy(),
    }


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
        "use_kmeans_init": True,
        "show_progress": True,
        "log_interval": 10,
        "print_config": True,
        "export_demo_json": True,
        "demo_json_output": None,
        "demo_proj_method": "pca",
        "demo_topk": 6,
        "demo_seed": 42,
        "save_demo_artifacts": False,
        "demo_artifacts_dir": "demo_data/artifacts",
        "demo_dataset_name": None,
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
    results = run_experiment(config)
    metrics = results["metrics"]

    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")
    print(f"ACC: {metrics['acc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")

    if bool(config.get("export_demo_json", True)):
        paths = resolve_dataset_pickle_paths(config["dataset_name"])
        display_dataset_name = config.get("demo_dataset_name") or to_display_dataset_name(paths["dataset_id"])

        default_output = Path("demo_data") / f"{display_dataset_name}.json"
        output_path_cfg = config.get("demo_json_output")
        output_path = Path(output_path_cfg) if output_path_cfg else default_output

        demo_result = convert_to_demo_json(
            dataset=display_dataset_name,
            features_path=paths["features"],
            hypergraph_path=paths["hypergraph"],
            final_emb=results["final_emb"],
            preds=results["preds"],
            output_path=output_path,
            labels=results["labels"],
            probs=results["probs"],
            metrics=metrics,
            topk=int(config.get("demo_topk", 6)),
            proj_method=str(config.get("demo_proj_method", "pca")),
            seed=int(config.get("demo_seed", 42)),
        )
        print(f"[Demo] Saved JSON: {output_path} (nodes={demo_result['num_nodes']}, clusters={demo_result['num_clusters']})")

        if bool(config.get("save_demo_artifacts", False)):
            artifacts_dir = Path(str(config.get("demo_artifacts_dir", "demo_data/artifacts")))
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            np.save(artifacts_dir / "final_emb.npy", results["final_emb"])
            np.save(artifacts_dir / "preds.npy", results["preds"])
            np.save(artifacts_dir / "probs.npy", results["probs"])
            with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"[Demo] Saved artifacts to: {artifacts_dir}")


if __name__ == "__main__":
    main()
