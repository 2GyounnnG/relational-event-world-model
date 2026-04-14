from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class SandboxLocalEventOperator(nn.Module):
    """Minimal masked local event operator for sandbox single-event samples."""

    def __init__(
        self,
        node_feature_dim: int = 7,
        edge_feature_dim: int = 4,
        event_param_dim: int = 4,
        num_event_types: int = 2,
        hidden_dim: int = 64,
        event_embed_dim: int = 16,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.event_param_dim = event_param_dim

        self.event_embedding = nn.Embedding(num_event_types, event_embed_dim)

        node_input_dim = node_feature_dim + event_embed_dim + event_param_dim + 2
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        edge_input_dim = edge_feature_dim + 2 * hidden_dim + event_embed_dim + event_param_dim + 2
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.edge_message = nn.Linear(hidden_dim, hidden_dim)
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_delta_head = nn.Linear(hidden_dim, node_feature_dim)
        self.edge_delta_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_feature_dim),
        )

    def forward(
        self,
        *,
        node_features_t: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features_t: torch.Tensor,
        event_type_id: torch.Tensor,
        event_params: torch.Tensor,
        event_node_mask: torch.Tensor,
        event_edge_mask: torch.Tensor,
        event_scope_node_mask: torch.Tensor,
        event_scope_edge_mask: torch.Tensor,
        node_batch_index: torch.Tensor,
        edge_batch_index: torch.Tensor,
        num_nodes_per_graph: torch.Tensor,
        num_edges_per_graph: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run one local message-passing/update step.

        Shapes:
            node_features_t: [N_total, F_node]
            edge_index: [2, E_total], already offset by the collate function
            edge_features_t: [E_total, F_edge]
            event_type_id: [B]
            event_params: [B, F_event]
            event_node_mask: [N_total], direct event target nodes
            event_edge_mask: [E_total], direct event target edges
            event_scope_node_mask: [N_total]
            event_scope_edge_mask: [E_total]
            node_batch_index: [N_total]
            edge_batch_index: [E_total]
            num_nodes_per_graph: [B]
            num_edges_per_graph: [B]
        """
        self._check_batch_counts(
            node_features_t=node_features_t,
            edge_features_t=edge_features_t,
            num_nodes_per_graph=num_nodes_per_graph,
            num_edges_per_graph=num_edges_per_graph,
        )

        node_scope = event_scope_node_mask.to(dtype=node_features_t.dtype).unsqueeze(-1)
        edge_scope = event_scope_edge_mask.to(dtype=edge_features_t.dtype).unsqueeze(-1)
        node_event_target = event_node_mask.to(dtype=node_features_t.dtype).unsqueeze(-1)
        edge_event_target = event_edge_mask.to(dtype=edge_features_t.dtype).unsqueeze(-1)

        graph_event = self.event_embedding(event_type_id)
        node_event = graph_event[node_batch_index]
        edge_event = graph_event[edge_batch_index]
        node_params = event_params[node_batch_index].to(dtype=node_features_t.dtype)
        edge_params = event_params[edge_batch_index].to(dtype=edge_features_t.dtype)

        node_context = torch.cat(
            [
                node_features_t,
                node_event.to(dtype=node_features_t.dtype),
                node_params,
                node_event_target,
                node_scope,
            ],
            dim=-1,
        )
        node_h = self.node_encoder(node_context)

        src, dst = edge_index[0], edge_index[1]
        edge_context = torch.cat(
            [
                edge_features_t,
                node_h[src],
                node_h[dst],
                edge_event.to(dtype=edge_features_t.dtype),
                edge_params,
                edge_event_target,
                edge_scope,
            ],
            dim=-1,
        )
        edge_h = self.edge_encoder(edge_context)

        # One local message pass: scoped edge messages are added to both spring
        # endpoints because generator edges represent undirected spring slots.
        messages = self.edge_message(edge_h) * edge_scope
        aggregated = torch.zeros_like(node_h)
        aggregated.index_add_(0, src, messages)
        aggregated.index_add_(0, dst, messages)

        node_h_updated = self.node_update(torch.cat([node_h, aggregated], dim=-1))
        raw_node_delta = self.node_delta_head(node_h_updated)
        raw_edge_delta = self.edge_delta_head(
            torch.cat([edge_h, node_h_updated[src], node_h_updated[dst]], dim=-1)
        )

        node_delta_pred = raw_node_delta * node_scope
        edge_delta_pred = raw_edge_delta * edge_scope

        node_scope_bool = node_scope > 0.5
        edge_scope_bool = edge_scope > 0.5
        node_features_pred = torch.where(
            node_scope_bool,
            node_features_t + node_delta_pred,
            node_features_t,
        )
        edge_features_pred = torch.where(
            edge_scope_bool,
            edge_features_t + edge_delta_pred,
            edge_features_t,
        )

        return {
            "node_delta_pred": node_delta_pred,
            "edge_delta_pred": edge_delta_pred,
            "node_features_pred": node_features_pred,
            "edge_features_pred": edge_features_pred,
        }

    @staticmethod
    def _check_batch_counts(
        *,
        node_features_t: torch.Tensor,
        edge_features_t: torch.Tensor,
        num_nodes_per_graph: torch.Tensor,
        num_edges_per_graph: torch.Tensor,
    ) -> None:
        if int(num_nodes_per_graph.sum().item()) != int(node_features_t.shape[0]):
            raise ValueError("num_nodes_per_graph does not match node_features_t")
        if int(num_edges_per_graph.sum().item()) != int(edge_features_t.shape[0]):
            raise ValueError("num_edges_per_graph does not match edge_features_t")


def _fake_tiny_batch() -> Dict[str, torch.Tensor]:
    node_features_t = torch.randn(5, 7)
    edge_features_t = torch.randn(4, 4)
    return {
        "node_features_t": node_features_t,
        "edge_index": torch.tensor(
            [
                [0, 1, 3, 3],
                [1, 2, 4, 4],
            ],
            dtype=torch.long,
        ),
        "edge_features_t": edge_features_t,
        "event_type_id": torch.tensor([0, 1], dtype=torch.long),
        "event_params": torch.tensor(
            [
                [0.1, -0.1, -1.0, -1.0],
                [0.0, 0.0, 3.0, 4.0],
            ],
            dtype=torch.float32,
        ),
        "event_node_mask": torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0]),
        "event_edge_mask": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "event_scope_node_mask": torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0]),
        "event_scope_edge_mask": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "node_batch_index": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        "edge_batch_index": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "num_nodes_per_graph": torch.tensor([3, 2], dtype=torch.long),
        "num_edges_per_graph": torch.tensor([2, 2], dtype=torch.long),
    }


def main() -> None:
    torch.manual_seed(0)
    batch = _fake_tiny_batch()
    model = SandboxLocalEventOperator()
    outputs = model(**batch)

    for key, value in outputs.items():
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")

    node_outside = batch["event_scope_node_mask"] <= 0.5
    edge_outside = batch["event_scope_edge_mask"] <= 0.5
    node_copy_exact = torch.equal(
        outputs["node_features_pred"][node_outside],
        batch["node_features_t"][node_outside],
    )
    edge_copy_exact = torch.equal(
        outputs["edge_features_pred"][edge_outside],
        batch["edge_features_t"][edge_outside],
    )
    print(f"outside_node_copy_exact={node_copy_exact}")
    print(f"outside_edge_copy_exact={edge_copy_exact}")


if __name__ == "__main__":
    main()
