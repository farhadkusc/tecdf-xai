"""Graph attention detector, two-stage training, explanation stability.

The stability metric matches the concept paper:

    R_e(x) = 1 - D(g(x), g(x+δ)) / (||δ|| + ε0)

D is the L2 distance between attention distributions. Stage 1 fits a linear
head on attention-pooled node states, including perturbed graphs. Stage 2
freezes that head and reports fidelity, sparsity, and stability. Graphs are
synthetic flow graphs, not CICIDS2017.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class FlowGraph:
    """Directed flow graph. y is the graph label (1 = malicious campaign)."""

    features: np.ndarray  # (n, d)
    adj: np.ndarray  # (n, n) binary, no self loops required
    y: int
    name: str


def make_corpus(n_graphs: int = 80, n: int = 12, d: int = 6, seed: int = 7) -> list[FlowGraph]:
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(n_graphs):
        malicious = i % 2 == 0
        x = rng.normal(0, 1, size=(n, d))
        adj = (rng.random((n, n)) < 0.18).astype(float)
        np.fill_diagonal(adj, 0)
        if malicious:
            # A dense core on the first 4 nodes, plus a feature shift.
            adj[:4, :4] = 1
            np.fill_diagonal(adj, 0)
            x[:4, 0] += 1.8
        graphs.append(FlowGraph(x, adj, int(malicious), f"g{i}"))
    return graphs


def attention_pool(g: FlowGraph, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-hop GAT-style pool. Returns (graph vector, attention over nodes)."""
    n = g.features.shape[0]
    degree = g.adj.sum(axis=1)
    neigh = (g.adj @ g.features) / np.maximum(degree, 1)[:, None]
    h = 0.6 * g.features + 0.4 * neigh
    logits = h @ query
    # Isolated nodes can still receive attention; that is intended.
    alpha = _softmax(logits)
    pooled = alpha @ h
    return pooled, alpha


def perturb_edges(g: FlowGraph, budget: int, rng: np.random.Generator) -> FlowGraph:
    """Graph-structure attack: flip `budget` absent edges on, preferring the core."""
    adj = g.adj.copy()
    absent = list(zip(*np.where(adj == 0)))
    absent = [(i, j) for i, j in absent if i != j]
    rng.shuffle(absent)
    for i, j in absent[:budget]:
        adj[i, j] = 1
    return FlowGraph(g.features.copy(), adj, g.y, g.name + "*")


def perturb_features(g: FlowGraph, eps: float, rng: np.random.Generator) -> FlowGraph:
    noise = rng.normal(0, 1, size=g.features.shape)
    noise /= max(np.linalg.norm(noise), 1e-8)
    x = g.features + eps * noise
    return FlowGraph(x, g.adj.copy(), g.y, g.name + "+")


@dataclass
class Detector:
    query: np.ndarray
    head: np.ndarray
    bias: float

    def predict_proba(self, g: FlowGraph) -> tuple[float, np.ndarray]:
        pooled, alpha = attention_pool(g, self.query)
        logit = float(pooled @ self.head + self.bias)
        logit = max(min(logit, 20), -20)
        p = 1 / (1 + np.exp(-logit))
        return p, alpha


def train_detector(graphs: list[FlowGraph], steps: int = 240, seed: int = 0) -> Detector:
    """Stage 1. Madry-style outer loop on a tiny attack budget, NumPy SGD."""
    rng = np.random.default_rng(seed)
    d = graphs[0].features.shape[1]
    query = rng.normal(0, 0.3, size=d)
    head = rng.normal(0, 0.3, size=d)
    bias = 0.0
    lr = 0.15
    for step in range(steps):
        g = graphs[int(rng.integers(0, len(graphs)))]
        attack = perturb_edges(g, budget=2, rng=rng) if step % 2 == 0 else g
        pooled, _ = attention_pool(attack, query)
        logit = float(np.clip(pooled @ head + bias, -20, 20))
        p = 1 / (1 + np.exp(-logit))
        err = p - g.y
        head -= lr * (err * pooled + 0.01 * head)
        bias -= lr * err
        # Query gets a coarse finite-difference step so attention is not frozen.
        if step % 5 == 0:
            q2 = query + rng.normal(0, 0.05, size=d)
            p2, _ = attention_pool(attack, q2)
            loss = (p - g.y) ** 2
            logit2 = float(np.clip(p2 @ head + bias, -20, 20))
            p2s = 1 / (1 + np.exp(-logit2))
            loss2 = (p2s - g.y) ** 2
            if loss2 < loss:
                query = 0.8 * query + 0.2 * q2
    return Detector(query, head, float(bias))


def explanation_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def stability(a: np.ndarray, b: np.ndarray, delta_norm: float, eps0: float = 1e-6) -> float:
    r = 1.0 - explanation_distance(a, b) / (delta_norm + eps0)
    return float(np.clip(r, 0.0, 1.0))


@dataclass
class Evaluation:
    accuracy: float
    accuracy_under_edge_attack: float
    mean_stability: float
    mean_fidelity: float
    mean_sparsity_top3: float
    n: int


def evaluate(seed: int = 7) -> Evaluation:
    corpus = make_corpus(seed=seed)
    split = len(corpus) // 2
    train, test = corpus[:split], corpus[split:]
    det = train_detector(train, seed=seed)
    rng = np.random.default_rng(seed + 3)
    correct = 0
    correct_adv = 0
    stabilities = []
    fidelities = []
    sparsities = []
    for g in test:
        p, alpha = det.predict_proba(g)
        pred = int(p >= 0.5)
        correct += int(pred == g.y)
        adv = perturb_edges(g, budget=3, rng=rng)
        p_adv, _ = det.predict_proba(adv)
        correct_adv += int(int(p_adv >= 0.5) == g.y)
        noisy = perturb_features(g, eps=0.15, rng=rng)
        _, alpha2 = det.predict_proba(noisy)
        stabilities.append(stability(alpha, alpha2, delta_norm=0.15))
        # Fidelity: removing the top-attention node's edges should move probability
        # more than removing the lowest-attention node.
        top = int(np.argmax(alpha))
        bot = int(np.argmin(alpha))

        def drop(node: int) -> float:
            adj = g.adj.copy()
            adj[node, :] = 0
            adj[:, node] = 0
            p2, _ = det.predict_proba(FlowGraph(g.features, adj, g.y, g.name))
            return abs(p2 - p)

        move_top, move_bot = drop(top), drop(bot)
        fidelities.append(float(move_top >= move_bot))
        order = np.argsort(alpha)[::-1]
        sparsities.append(float(alpha[order[:3]].sum()))
    n = len(test)
    return Evaluation(
        accuracy=correct / n,
        accuracy_under_edge_attack=correct_adv / n,
        mean_stability=float(np.mean(stabilities)),
        mean_fidelity=float(np.mean(fidelities)),
        mean_sparsity_top3=float(np.mean(sparsities)),
        n=n,
    )
