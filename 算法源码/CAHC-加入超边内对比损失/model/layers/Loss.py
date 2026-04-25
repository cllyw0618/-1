import torch
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
