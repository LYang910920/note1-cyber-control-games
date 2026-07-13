"""State-conditioned actor and critic architectures for Note 1."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cybercontrol.nn import (
    ArchitectureDescriptor,
    ArchitectureRegistry,
    DenseGraphEncoder,
    MLP,
    parameter_count,
)


class BudgetedCommunityActor(nn.Module):
    """Shared local encoder with one globally budgeted intervention choice.

    For ``M`` communities, the categorical support contains one no-op plus
    ``M x 2`` community/mode choices (patch or clean). A sampled joint action
    therefore activates at most one community and is directly comparable with
    the budget-matched heuristic baselines.
    """

    def __init__(self, observation_dim: int, hidden: int = 64):
        super().__init__()
        self.local = MLP(observation_dim, 2, width=hidden, depth=2, activation="tanh")
        self.no_op_logit = nn.Parameter(torch.zeros(()))

    def forward(self, observations, adjacency=None):
        del adjacency
        squeeze = observations.ndim == 2
        if squeeze:
            observations = observations.unsqueeze(0)
        local_logits = self.local(observations).flatten(start_dim=1)
        no_op = self.no_op_logit.expand(observations.shape[0], 1)
        logits = torch.cat([no_op, local_logits], dim=1)
        return logits.squeeze(0) if squeeze else logits

    @staticmethod
    def decode(index: int, communities: int) -> np.ndarray:
        actions = np.zeros(communities, dtype=np.int64)
        if int(index) > 0:
            encoded = int(index) - 1
            community = encoded // 2
            mode = encoded % 2 + 1
            actions[community] = mode
        return actions

    def greedy(
        self,
        observations: np.ndarray,
        device: str,
        adjacency: np.ndarray | None = None,
    ) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            index = int(torch.argmax(self(tensor, adjacency)).item())
        return self.decode(index, observations.shape[0])


class PooledStateCritic(nn.Module):
    """Permutation-aware value baseline over community observations."""

    def __init__(self, observation_dim: int, hidden: int = 64):
        super().__init__()
        self.local = MLP(observation_dim, hidden, width=hidden, depth=2, activation="tanh")
        self.value = MLP(hidden, 1, width=hidden, depth=2, activation="tanh")

    def forward(self, observations, adjacency=None):
        del adjacency
        if observations.ndim == 2:
            observations = observations.unsqueeze(0)
        pooled = self.local(observations).mean(dim=1)
        return self.value(pooled).squeeze(-1)


class GraphBudgetedCommunityActor(BudgetedCommunityActor):
    """Graph-context actor with shared node weights and one joint action budget."""

    def __init__(self, observation_dim: int, hidden: int = 48, graph_layers: int = 1):
        nn.Module.__init__(self)
        self.local = MLP(
            observation_dim,
            hidden,
            width=hidden,
            depth=2,
            activation="tanh",
        )
        self.graph = DenseGraphEncoder(
            hidden,
            hidden=hidden,
            layers=graph_layers,
            activation="tanh",
        )
        self.mode = nn.Linear(hidden, 2)
        self.no_op_logit = nn.Parameter(torch.zeros(()))

    def forward(self, observations, adjacency=None):
        if adjacency is None:
            raise ValueError("graph actor requires a community adjacency matrix")
        squeeze = observations.ndim == 2
        if squeeze:
            observations = observations.unsqueeze(0)
        hidden = self.local(observations)
        adjacency = torch.as_tensor(
            adjacency,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        hidden = self.graph(hidden, adjacency)
        local_logits = self.mode(hidden).flatten(start_dim=1)
        no_op = self.no_op_logit.expand(observations.shape[0], 1)
        logits = torch.cat([no_op, local_logits], dim=1)
        return logits.squeeze(0) if squeeze else logits


class GraphPooledStateCritic(nn.Module):
    """Centralized graph encoder followed by permutation-invariant pooling."""

    def __init__(self, observation_dim: int, hidden: int = 48, graph_layers: int = 1):
        super().__init__()
        self.local = MLP(
            observation_dim,
            hidden,
            width=hidden,
            depth=2,
            activation="tanh",
        )
        self.graph = DenseGraphEncoder(
            hidden,
            hidden=hidden,
            layers=graph_layers,
            activation="tanh",
        )
        self.value = MLP(hidden, 1, width=hidden, depth=2, activation="tanh")

    def forward(self, observations, adjacency=None):
        if adjacency is None:
            raise ValueError("graph critic requires a community adjacency matrix")
        if observations.ndim == 2:
            observations = observations.unsqueeze(0)
        hidden = self.local(observations)
        adjacency = torch.as_tensor(
            adjacency,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        encoded = self.graph(hidden, adjacency)
        return self.value(encoded.mean(dim=1)).squeeze(-1)


def _build_summary_pair(observation_dim: int, hidden: int, graph_layers: int = 1):
    del graph_layers
    return {
        "actor": BudgetedCommunityActor(observation_dim, hidden),
        "critic": PooledStateCritic(observation_dim, hidden),
    }


def _build_graph_pair(observation_dim: int, hidden: int, graph_layers: int = 1):
    return {
        "actor": GraphBudgetedCommunityActor(observation_dim, hidden, graph_layers),
        "critic": GraphPooledStateCritic(observation_dim, hidden, graph_layers),
    }


ARCHITECTURE_REGISTRY = ArchitectureRegistry()
ARCHITECTURE_REGISTRY.register(
    ArchitectureDescriptor(
        name="summary_mlp",
        input_shape="[batch, communities, observation_dim]",
        output_shape="actor [batch, 1+2*communities]; critic [batch]",
        activation="tanh",
        normalization="none",
        encoder="shared community MLP",
        pooling="mean for centralized value critic",
        decoder="budgeted categorical mode head and scalar value head",
    ),
    _build_summary_pair,
)
ARCHITECTURE_REGISTRY.register(
    ArchitectureDescriptor(
        name="graph_context",
        input_shape="[batch, communities, observation_dim] plus [communities, communities]",
        output_shape="actor [batch, 1+2*communities]; critic [batch]",
        activation="tanh",
        normalization="row-normalized community adjacency",
        encoder="shared MLP plus dense graph message passing",
        pooling="mean after graph encoder for centralized value critic",
        decoder="budgeted categorical mode head and scalar value head",
    ),
    _build_graph_pair,
)


def matched_graph_mappo_width(
    observation_dim: int,
    baseline_hidden: int,
    *,
    graph_layers: int = 1,
) -> tuple[int, int, int]:
    """Choose a graph actor/critic width closest to the summary-MLP budget."""

    baseline = ARCHITECTURE_REGISTRY.build("summary_mlp", observation_dim, baseline_hidden)
    target = int(ARCHITECTURE_REGISTRY.describe("summary_mlp", baseline)["parameters"])
    candidates = []
    for width in range(4, baseline_hidden + 1):
        graph = ARCHITECTURE_REGISTRY.build(
            "graph_context",
            observation_dim,
            width,
            graph_layers,
        )
        count = int(ARCHITECTURE_REGISTRY.describe("graph_context", graph)["parameters"])
        candidates.append((abs(count - target), width, count))
    _, width, count = min(candidates)
    return width, target, count


class StateConditionedCommunityPolicy(nn.Module):
    """Permutation-equivariant community score policy for one player role."""

    def __init__(self, observation_dim: int, hidden: int = 64):
        super().__init__()
        self.local = MLP(observation_dim, hidden, width=hidden, depth=2, activation="tanh")
        self.context = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, observations):
        local = self.local(observations)
        context = self.context(local.mean(dim=-2, keepdim=True))
        return self.score(torch.tanh(local + context)).squeeze(-1)


def architecture_record(
    name: str, model: nn.Module, **dimensions: int | str
) -> dict[str, int | str]:
    """Return a table-ready architecture record."""

    return {"name": name, "parameters": parameter_count(model), **dimensions}
