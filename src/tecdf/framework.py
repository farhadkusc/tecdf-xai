"""Service-neighborhood detector and a numeric logistic baseline on NSL-KDD.

The graph is the service co-membership graph: a flow's neighbors are the
training flows with the same service. The graph score is that neighborhood's
attack rate, computed on the training split only.

A graph-structure attack rewrites the service of a test flow, which moves it
into a different neighborhood. Explanation stability perturbs the numeric
features and measures how far the attribution vector moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nsl import NUMERIC, load


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def _fit(X: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.4, l2: float = 0.01):
    mu = X.mean(0)
    sd = np.where(X.std(0) < 1e-6, 1.0, X.std(0))
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = 0.0
    y = y.astype(float)
    for _ in range(steps):
        p = _sigmoid(Z @ w + b)
        err = p - y
        w -= lr * (Z.T @ err / len(y) + l2 * w)
        b -= lr * float(err.mean())
    return w, b, mu, sd


def _predict(X, w, b, mu, sd) -> np.ndarray:
    return _sigmoid((X - mu) / sd @ w + b)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    order = np.argsort(-p, kind="mergesort")
    rel = y[order].astype(float)
    ap = float(np.sum((np.cumsum(rel) / np.arange(1, len(rel) + 1)) * rel) / max(rel.sum(), 1))
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "average_precision": round(ap, 4),
    }


def _split(n: int, y: np.ndarray, seed: int = 7, test_frac: float = 0.3):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for label in (0, 1):
        pool = np.flatnonzero(y == label)
        rng.shuffle(pool)
        n_test = int(round(len(pool) * test_frac))
        test_idx.append(pool[:n_test])
        train_idx.append(pool[n_test:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def _service_rate(services: np.ndarray, y: np.ndarray) -> dict[str, float]:
    total: dict[str, int] = {}
    attacks: dict[str, int] = {}
    for s, label in zip(services, y):
        total[s] = total.get(s, 0) + 1
        attacks[s] = attacks.get(s, 0) + int(label)
    return {s: attacks[s] / total[s] for s in total}


@dataclass
class Report:
    n_train: int
    n_test: int
    graph: dict
    logistic: dict
    graph_under_service_rewire: dict
    stability_mean: float
    feature_names: list[str]


def evaluate(seed: int = 7, attack_frac: float = 0.2) -> dict:
    x, y, services = load()
    train, test = _split(len(y), y, seed=seed)
    attack_rate = float(y[test].mean())
    majority_accuracy = max(attack_rate, 1.0 - attack_rate)
    rate = _service_rate(services[train], y[train])
    global_rate = float(y[train].mean())
    graph_p = np.array([rate.get(services[i], global_rate) for i in test])
    w, b, mu, sd = _fit(x[train], y[train])
    logit_p = _predict(x[test], w, b, mu, sd)

    rng = np.random.default_rng(seed + 1)
    attacked = services.copy()
    catalog = np.array(sorted(rate))
    n_flip = int(round(attack_frac * len(test)))
    chosen = rng.choice(test, size=n_flip, replace=False)
    for i in chosen:
        current = services[i]
        options = catalog[catalog != current]
        attacked[i] = options[int(rng.integers(0, len(options)))]
    attacked_p = np.array([rate.get(attacked[i], global_rate) for i in test])

    # Attribution is the standardized feature times its weight.
    Z = (x[test] - mu) / sd
    attr = Z * w
    noise = rng.normal(0, 0.15, size=x[test].shape)
    Z2 = ((x[test] + noise) - mu) / sd
    attr2 = Z2 * w
    delta = np.linalg.norm(noise, axis=1) + 1e-6
    num = np.sum(attr * attr2, axis=1)
    den = np.linalg.norm(attr, axis=1) * np.linalg.norm(attr2, axis=1) + 1e-8
    cosine = np.clip(num / den, -1.0, 1.0)
    # Paper-form ratio, clipped. D is the L2 gap of the attribution vectors.
    paper_r = np.clip(1.0 - np.linalg.norm(attr - attr2, axis=1) / delta, 0.0, 1.0)

    return {
        "dataset": "NSL-KDD KDDTrain+_20Percent",
        "source": "https://github.com/tuanio/nsl_kdd (20% training split of the UNB NSL-KDD release)",
        "citation": "Tavallaee, Bagheri, Lu, Ghorbani. A Detailed Analysis of the KDD CUP 99 Data Set. IEEE CISDA 2009.",
        "split": "stratified 70/30, seed 7, training neighborhoods only",
        "n_rows": int(len(y)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_attack_rate": round(attack_rate, 4),
        "majority_class_accuracy": round(majority_accuracy, 4),
        "graph_service_neighborhood": _metrics(y[test], graph_p),
        "numeric_logistic": _metrics(y[test], logit_p),
        "graph_after_service_rewire": {
            **_metrics(y[test], attacked_p),
            "fraction_rewired": attack_frac,
        },
        "explanation_cosine_stability": round(float(cosine.mean()), 4),
        "explanation_paper_R_e": round(float(paper_r.mean()), 4),
        "feature_names": list(NUMERIC),
        "logistic_weights_standardized": {name: round(float(coef), 4) for name, coef in zip(NUMERIC, w)},
        "disclaimer": "One public split. Not CICIDS2017 and not a human analyst study.",
    }
