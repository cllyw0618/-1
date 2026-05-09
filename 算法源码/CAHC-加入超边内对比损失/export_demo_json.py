import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


PRIVACY_NOTE = "根据数据合规与匿名化展示原则，系统使用节点编号代替原始对象名称。"


def to_numpy(data: Any) -> np.ndarray:
    if data is None:
        raise ValueError("Input data is None.")
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    elif hasattr(data, "toarray"):
        data = data.toarray()
    return np.asarray(data)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
    elif path.suffix.lower() in {".pickle", ".pkl"}:
        arr = load_pickle(path)
    else:
        raise ValueError(f"Unsupported array file: {path}")
    return to_numpy(arr)


def format_node_name(dataset: str, node_id: int) -> str:
    return f"{dataset} Node #{node_id:04d}"


def hungarian_mapping(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    d = int(max(y_true.max(), y_pred.max()) + 1)
    weight = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        weight[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(weight.max() - weight)
    return weight, row_ind, col_ind


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    weight, row_ind, col_ind = hungarian_mapping(y_true, y_pred)
    acc = float(weight[row_ind, col_ind].sum() / y_pred.size)

    mapped_pred = np.zeros_like(y_pred)
    for i, cluster_id in enumerate(row_ind):
        mapped_pred[y_pred == cluster_id] = col_ind[i]
    f1 = float(f1_score(y_true, mapped_pred, average="macro"))

    return {
        "acc": acc,
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "f1": f1,
    }


def normalize_to_0_100(coords_2d: np.ndarray) -> np.ndarray:
    coords = coords_2d.astype(np.float64, copy=True)
    for col in range(2):
        mn = float(coords[:, col].min())
        mx = float(coords[:, col].max())
        if mx - mn < 1e-12:
            coords[:, col] = 50.0
        else:
            coords[:, col] = (coords[:, col] - mn) / (mx - mn) * 100.0
    return coords


def project_2d(emb: np.ndarray, method: str = "pca", seed: int = 42) -> np.ndarray:
    method = method.lower()
    if emb.shape[0] == 1:
        return np.array([[50.0, 50.0]], dtype=np.float64)

    if method == "pca":
        reduced = PCA(n_components=2, random_state=seed).fit_transform(emb)
    elif method == "tsne":
        perplexity = min(30.0, max(5.0, float((emb.shape[0] - 1) // 3)))
        reduced = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(emb)
    elif method == "umap":
        try:
            import umap  # type: ignore
        except Exception:
            reduced = PCA(n_components=2, random_state=seed).fit_transform(emb)
        else:
            reducer = umap.UMAP(n_components=2, random_state=seed)
            reduced = reducer.fit_transform(emb)
    else:
        raise ValueError(f"Unsupported projection method: {method}")

    return normalize_to_0_100(reduced)


def safe_int(x: Any) -> Optional[int]:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return None


def infer_edge_to_nodes(hypergraph: Dict[Any, Any], num_nodes: int) -> List[List[int]]:
    keys = list(hypergraph.keys())
    key_ids = [safe_int(k) for k in keys]
    key_numeric = [k for k in key_ids if k is not None]
    key_in_range = sum(1 for k in key_numeric if 0 <= k < num_nodes) / max(len(key_numeric), 1)

    value_sample: List[Any] = []
    for v in hypergraph.values():
        if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
            value_sample.extend(list(v))
        if len(value_sample) >= 50000:
            break

    val_ids = [safe_int(x) for x in value_sample]
    val_numeric = [v for v in val_ids if v is not None]
    val_in_range = sum(1 for v in val_numeric if 0 <= v < num_nodes) / max(len(val_numeric), 1)

    assume_edge_to_nodes = val_in_range >= key_in_range
    edge_to_nodes: List[List[int]] = []

    if assume_edge_to_nodes:
        for nodes in hypergraph.values():
            node_ids = []
            for nid in nodes:
                nid_i = safe_int(nid)
                if nid_i is not None and 0 <= nid_i < num_nodes:
                    node_ids.append(nid_i)
            if len(node_ids) > 1:
                edge_to_nodes.append(node_ids)
        return edge_to_nodes

    edge_map: Dict[Any, List[int]] = defaultdict(list)
    for node_key, edges in hypergraph.items():
        node_id = safe_int(node_key)
        if node_id is None or not (0 <= node_id < num_nodes):
            continue
        for edge_id in edges:
            parsed = safe_int(edge_id)
            edge_key = parsed if parsed is not None else str(edge_id)
            edge_map[edge_key].append(node_id)

    for nodes in edge_map.values():
        if len(nodes) > 1:
            edge_to_nodes.append(nodes)
    return edge_to_nodes


def build_shared_hyperedge_neighbors(edge_to_nodes: List[List[int]], num_nodes: int, topk: int) -> List[List[Tuple[int, int]]]:
    shared = [Counter() for _ in range(num_nodes)]
    for nodes in edge_to_nodes:
        unique_nodes = sorted(set(nodes))
        for u in unique_nodes:
            for v in unique_nodes:
                if u != v:
                    shared[u][v] += 1

    result: List[List[Tuple[int, int]]] = []
    for i in range(num_nodes):
        top_neighbors = shared[i].most_common(topk)
        result.append([(nid, int(cnt)) for nid, cnt in top_neighbors])
    return result


def build_topk_cosine_neighbors(emb: np.ndarray, topk: int, block_size: int = 1024) -> List[List[Tuple[int, float]]]:
    n = emb.shape[0]
    if n == 0:
        return []
    if topk <= 0:
        return [[] for _ in range(n)]
    if n == 1:
        return [[]]

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normed = emb / norms

    neighbors: List[List[Tuple[int, float]]] = []
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        sims = normed[start:end] @ normed.T
        row_idx = np.arange(start, end)
        sims[np.arange(end - start), row_idx] = -np.inf

        kk = min(topk, n - 1)
        idx_part = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
        score_part = np.take_along_axis(sims, idx_part, axis=1)

        order = np.argsort(-score_part, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)
        score_sorted = np.take_along_axis(score_part, order, axis=1)

        for r in range(end - start):
            row_neighbors = [
                (int(idx_sorted[r, c]), float(score_sorted[r, c]))
                for c in range(kk)
                if np.isfinite(score_sorted[r, c])
            ]
            neighbors.append(row_neighbors)
    return neighbors


def normalize_metrics_keys(metrics: Dict[str, Any]) -> Dict[str, Any]:
    canonical = {"acc": None, "nmi": None, "ari": None, "f1": None}
    aliases = {
        "acc": "acc",
        "accuracy": "acc",
        "nmi": "nmi",
        "ari": "ari",
        "f1": "f1",
    }
    for k, v in metrics.items():
        lk = str(k).strip().lower()
        if lk in aliases:
            canonical[aliases[lk]] = v
    return {k: v for k, v in canonical.items() if v is not None}


def compute_cluster_purity(preds: np.ndarray, labels: np.ndarray, cluster_id: int) -> Optional[float]:
    idx = np.where(preds == cluster_id)[0]
    if idx.size == 0:
        return None
    cluster_labels = labels[idx]
    counts = Counter(cluster_labels.tolist())
    return float(max(counts.values()) / idx.size)


def convert_to_demo_json(
    dataset: str,
    features_path: Path,
    hypergraph_path: Path,
    final_emb: np.ndarray,
    preds: np.ndarray,
    output_path: Path,
    labels: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, float]] = None,
    topk: int = 6,
    proj_method: str = "pca",
    seed: int = 42,
) -> Dict[str, Any]:
    features = load_pickle(features_path)
    hypergraph = load_pickle(hypergraph_path)

    if not hasattr(features, "shape"):
        raise ValueError("features.pickle must contain an array-like object with .shape")
    if not isinstance(hypergraph, dict):
        raise ValueError("hypergraph.pickle must be a dict")

    final_emb = to_numpy(final_emb)
    preds = to_numpy(preds).reshape(-1).astype(np.int64)
    labels = None if labels is None else to_numpy(labels).reshape(-1).astype(np.int64)
    probs = None if probs is None else to_numpy(probs)

    num_nodes = int(final_emb.shape[0])
    if preds.shape[0] != num_nodes:
        raise ValueError(f"preds length ({preds.shape[0]}) != num_nodes ({num_nodes})")
    if getattr(features, "shape", (None,))[0] is not None and int(features.shape[0]) != num_nodes:
        raise ValueError(f"features rows ({features.shape[0]}) != num_nodes ({num_nodes})")
    if labels is not None and labels.shape[0] != num_nodes:
        raise ValueError(f"labels length ({labels.shape[0]}) != num_nodes ({num_nodes})")
    if probs is not None and probs.shape[0] != num_nodes:
        raise ValueError(f"probs rows ({probs.shape[0]}) != num_nodes ({num_nodes})")

    num_clusters = int(probs.shape[1]) if probs is not None and probs.ndim == 2 else int(preds.max() + 1)
    coords = project_2d(final_emb, method=proj_method, seed=seed)
    top_sim = build_topk_cosine_neighbors(final_emb, topk=topk)

    edge_to_nodes = infer_edge_to_nodes(hypergraph, num_nodes=num_nodes)
    shared_neighbors = build_shared_hyperedge_neighbors(edge_to_nodes, num_nodes=num_nodes, topk=topk)

    if metrics is None:
        if labels is not None:
            metrics = calculate_metrics(labels, preds)
        else:
            metrics = {}

    clusters: List[Dict[str, Any]] = []
    for cid in range(num_clusters):
        count = int(np.sum(preds == cid))
        cluster_item: Dict[str, Any] = {
            "id": cid,
            "name": f"Cluster {cid}",
            "count": count,
        }
        if labels is not None:
            purity = compute_cluster_purity(preds, labels, cid)
            if purity is not None:
                cluster_item["purity"] = purity
        clusters.append(cluster_item)

    node_items: List[Dict[str, Any]] = []
    for i in range(num_nodes):
        confidence = None
        if probs is not None and probs.ndim == 2 and int(preds[i]) < probs.shape[1]:
            confidence = float(probs[i, int(preds[i])])

        sim_items = []
        for nid, score in top_sim[i]:
            reason = "嵌入空间距离较近"
            if int(preds[nid]) == int(preds[i]):
                reason = "嵌入空间距离较近，且属于同一预测簇"
            sim_items.append(
                {
                    "id": nid,
                    "name": format_node_name(dataset, nid),
                    "cluster": int(preds[nid]),
                    "score": score,
                    "reason": reason,
                }
            )

        hyperedge_items = []
        for nid, cnt in shared_neighbors[i]:
            hyperedge_items.append(
                {
                    "id": nid,
                    "name": format_node_name(dataset, nid),
                    "cluster": int(preds[nid]),
                    "shared_hyperedges": cnt,
                    "reason": "与当前节点共享高阶超边关系",
                }
            )

        node_items.append(
            {
                "id": i,
                "name": format_node_name(dataset, i),
                "cluster": int(preds[i]),
                "true_label": None if labels is None else int(labels[i]),
                "confidence": confidence,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "top_similar": sim_items,
                "hyperedge_neighbors": hyperedge_items,
            }
        )

    output = {
        "dataset": dataset,
        "node_type": "匿名样本节点",
        "num_nodes": num_nodes,
        "num_clusters": num_clusters,
        "privacy_note": PRIVACY_NOTE,
        "metrics": metrics,
        "clusters": clusters,
        "nodes": node_items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert trained hypergraph clustering outputs to frontend demo JSON.")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--features", type=Path, required=True, help="Path to features.pickle")
    parser.add_argument("--hypergraph", type=Path, required=True, help="Path to hypergraph.pickle")
    parser.add_argument("--labels", type=Path, default=None, help="Path to labels.pickle (optional)")
    parser.add_argument("--final-emb", type=Path, required=True, help="Path to final_emb.npy or pickle")
    parser.add_argument("--preds", type=Path, required=True, help="Path to preds.npy or pickle")
    parser.add_argument("--probs", type=Path, default=None, help="Path to probs.npy or pickle (optional)")
    parser.add_argument("--metrics", type=Path, default=None, help="Path to metrics.json (optional)")
    parser.add_argument("--output", type=Path, default=Path("demo_data/Cora.json"))
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--proj-method", type=str, default="pca", choices=["pca", "tsne", "umap"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    final_emb = load_array(args.final_emb)
    preds = load_array(args.preds)
    labels = load_array(args.labels) if args.labels else None
    probs = load_array(args.probs) if args.probs else None

    metrics = None
    if args.metrics:
        with args.metrics.open("r", encoding="utf-8") as f:
            metrics = normalize_metrics_keys(json.load(f))

    result = convert_to_demo_json(
        dataset=args.dataset,
        features_path=args.features,
        hypergraph_path=args.hypergraph,
        final_emb=final_emb,
        preds=preds,
        output_path=args.output,
        labels=labels,
        probs=probs,
        metrics=metrics,
        topk=args.topk,
        proj_method=args.proj_method,
        seed=args.seed,
    )

    print(f"Saved: {args.output}")
    print(f"num_nodes={result['num_nodes']}, num_clusters={result['num_clusters']}")
    if result.get("metrics"):
        print("metrics:", ", ".join([f"{k}={v:.4f}" for k, v in result["metrics"].items() if isinstance(v, (int, float))]))


if __name__ == "__main__":
    main()
