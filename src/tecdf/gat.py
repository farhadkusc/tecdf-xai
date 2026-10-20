"""Feature attention over the service neighborhood.

alpha_j = softmax(h_i · h_j / sqrt(d)) on a sample of training flows that
share the service. The pooled vector is the attention-weighted neighbor
features. A second scalar is the attention-weighted training label. The
output logistic is fit on the flow's own features, the pooled features, and
that scalar. Test labels are not in the neighborhood.

R_e uses N(0, 0.15) feature noise, the same scale as the logistic attribution
check, and only rows whose predicted class does not change. Attention entropy
is reported next to the entropy of a uniform distribution so a collapsed
attention vector is visible.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def _softmax(scores):
    m = scores.max(axis=1, keepdims=True)
    w = np.exp(np.clip(scores - m, -40, 40))
    w /= w.sum(axis=1, keepdims=True)
    return w


def _neighbor_index(services, train, k, rng):
    pools = {}
    for i in train:
        pools.setdefault(services[i], []).append(int(i))
    pools = {key: np.asarray(value, dtype=int) for key, value in pools.items()}
    catalog = np.array(sorted(pools))
    out = np.empty((len(services), k), dtype=int)
    for i in range(len(services)):
        pool = pools.get(services[i])
        if pool is None or len(pool) == 0:
            pool = pools[catalog[0]]
        pool = pool[pool != i]
        if len(pool) == 0:
            pool = pools.get(services[i], pools[catalog[0]])
        take = pool if len(pool) <= k else pool[rng.choice(len(pool), size=k, replace=False)]
        if len(take) < k:
            take = np.resize(take, k)
        out[i] = take
    return out


def _fit_logistic(features, y, steps=250, lr=0.35, l2=0.01):
    mu = features.mean(0)
    sd = np.where(features.std(0) < 1e-6, 1.0, features.std(0))
    z = (features - mu) / sd
    w = np.zeros(z.shape[1])
    b = float(np.log(max(y.mean(), 1e-4) / max(1.0 - y.mean(), 1e-4)))
    for _ in range(steps):
        p = _sigmoid(z @ w + b)
        err = p - y
        w -= lr * (z.T @ err / len(y) + l2 * w)
        b -= lr * float(err.mean())
    return w, b, mu, sd


def _attend(z, y, idx, neighbors):
    query = z[idx]
    neigh = z[neighbors[idx]]
    labels = y[neighbors[idx]]
    scores = np.einsum("md,mkd->mk", query, neigh) / np.sqrt(z.shape[1])
    alpha = _softmax(scores)
    pooled = np.einsum("mk,mkd->md", alpha, neigh)
    label = np.sum(alpha * labels, axis=1)
    entropy = -np.sum(alpha * np.log(alpha + 1e-12), axis=1)
    return pooled, label, alpha, entropy


def fit_gat(x, y, services, train, test, seed=7, k=16):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = x[train].mean(0)
    sd = np.where(x[train].std(0) < 1e-6, 1.0, x[train].std(0))
    z = (x - mu) / sd
    neighbors = _neighbor_index(services, np.asarray(train, dtype=int), k, rng)
    train = np.asarray(train, dtype=int)
    test = np.asarray(test, dtype=int)
    pooled, label, alpha, entropy = _attend(z, y, train, neighbors)
    features = np.concatenate([z[train], pooled, label[:, None]], axis=1)
    w, b, fmu, fsd = _fit_logistic(features, y[train])

    def predict(idx, nb, features_z):
        pooled_i, label_i, alpha_i, entropy_i = _attend(features_z, y, idx, nb)
        block = np.concatenate([features_z[idx], pooled_i, label_i[:, None]], axis=1)
        p = _sigmoid((block - fmu) / fsd @ w + b)
        return p, alpha_i, entropy_i

    p_test, alpha_test, entropy_test = predict(test, neighbors, z)
    noise = rng.normal(0.0, 0.15, size=z.shape)
    p_noisy, alpha_noisy, _entropy_noisy = predict(test, neighbors, z + noise)
    same = (p_test >= 0.5) == (p_noisy >= 0.5)
    if int(same.sum()) == 0:
        paper_r = 0.0
    else:
        gap = np.linalg.norm(alpha_test[same] - alpha_noisy[same], axis=1)
        delta = np.linalg.norm(noise[test][same], axis=1) + 1e-6
        paper_r = float(np.clip(1.0 - gap / delta, 0.0, 1.0).mean())

    attacked = services.copy()
    catalog = np.array(sorted({services[i] for i in train}))
    for i in test:
        choices = catalog[catalog != services[i]]
        if len(choices):
            attacked[i] = choices[int(rng.integers(0, len(choices)))]
    attacked_neighbors = _neighbor_index(attacked, train, k, rng)
    p_adv, _a, _e = predict(test, attacked_neighbors, z)
    uniform = float(np.log(k))
    return {
        "p_test": p_test,
        "p_adv": p_adv,
        "paper_r_e": paper_r,
        "class_unchanged_fraction": float(same.mean()),
        "attention_entropy": float(entropy_test.mean()),
        "uniform_entropy": uniform,
        "attention_l2": float(np.linalg.norm(alpha_test, axis=1).mean()),
    }
