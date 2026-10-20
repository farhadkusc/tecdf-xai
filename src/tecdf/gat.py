"""One-layer graph attention detector.

h_v = tanh(W x_v)
alpha_uv = softmax(LeakyReLU(a · h_u)) over a sampled same-service neighborhood
z_v = sum_u alpha_uv W_n h_u
The output logistic is trained with gradient steps. One inner step rewires the
neighborhood and keeps the step when the loss rises. R_e uses the L2 gap of
the attention rows on perturbations that leave the predicted class unchanged.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def _softmax(scores):
    m = scores.max(axis=1, keepdims=True)
    w = np.exp(scores - m)
    w /= w.sum(axis=1, keepdims=True)
    return w


def _neighbors(services: np.ndarray, train: np.ndarray, k: int, rng) -> np.ndarray:
    by = {}
    for i in train:
        by.setdefault(services[i], []).append(int(i))
    pools = {s: np.asarray(ix, dtype=int) for s, ix in by.items()}
    out = np.zeros((len(services), k), dtype=int)
    for i in range(len(services)):
        pool = pools.get(services[i], train)
        if len(pool) == 1:
            out[i] = pool[0]
            continue
        take = pool[rng.integers(0, len(pool), size=k)]
        out[i] = take
    return out


def fit_gat(x, y, services, train, test, seed=7, k=8, dim=8, epochs=6):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = x[train].mean(0)
    sd = np.where(x[train].std(0) < 1e-6, 1.0, x[train].std(0))
    Z = (x - mu) / sd
    W = rng.normal(0, 0.2, size=(Z.shape[1], dim))
    Wn = rng.normal(0, 0.2, size=(dim, dim))
    a = rng.normal(0, 0.1, size=dim)
    w = np.zeros(dim)
    s = 1.0
    b = float(np.log(y[train].mean() / max(1.0 - y[train].mean(), 1e-6)))
    neigh = _neighbors(services, train, k, rng)
    # A second neighborhood is the inner-max graph edit: same size, other services.
    other = services.copy()
    catalog = np.array(sorted(set(services[train].tolist())))
    for i in train:
        choices = catalog[catalog != services[i]]
        if len(choices):
            other[i] = choices[int(rng.integers(0, len(choices)))]
    neigh_adv = _neighbors(other, train, k, rng)

    def forward(idx, nb):
        H = np.tanh(Z @ W)
        Hn = np.tanh(H[nb[idx]] @ Wn)
        alpha = _softmax(Hn @ a)
        out = np.sum(alpha[:, :, None] * Hn, axis=1)
        neigh_y = y[nb[idx]]
        label_pool = np.sum(alpha * neigh_y, axis=1)
        logit = out @ w + s * label_pool + b
        return Hn, alpha, out, label_pool, logit

    idx_all = np.asarray(train, dtype=int)
    for _ in range(epochs):
        rng.shuffle(idx_all)
        for start in range(0, len(idx_all), 2048):
            batch = idx_all[start : start + 2048]
            use_adv = _loss_higher(forward, batch, neigh, neigh_adv, y)
            nb = neigh_adv if use_adv else neigh
            Hn, alpha, out, label_pool, logit = forward(batch, nb)
            p = _sigmoid(logit)
            err = p - y[batch]
            n = len(batch)
            w -= 0.3 * (out.T @ err / n + 0.01 * w)
            s -= 0.3 * float((err * label_pool).mean())
            b -= 0.3 * float(err.mean())
            dout = err[:, None] * w
            dalpha = np.sum(dout[:, None, :] * Hn, axis=2) + err[:, None] * s * y[nb[batch]]
            dot = np.sum(alpha * dalpha, axis=1, keepdims=True)
            dscore = alpha * (dalpha - dot)
            a -= 0.1 * (np.sum(dscore[:, :, None] * Hn, axis=(0, 1)) / n)
            a *= 0.999

    def predict(idx):
        _Hn, alpha, _out, _pool, logit = forward(np.asarray(idx, dtype=int), neigh)
        return _sigmoid(logit), alpha

    p_test, alpha_test = predict(test)
    # Explanation attack: perturb inputs, keep rows whose class does not flip.
    noise = rng.normal(0, 0.05, size=Z.shape)
    Z_saved = Z
    # local forward on perturbed features
    Z = Z_saved + noise
    p2, alpha2 = predict(test)
    Z = Z_saved
    same = ((p_test >= 0.5) == (p2 >= 0.5))
    if int(same.sum()) == 0:
        paper_r = 0.0
    else:
        delta = np.linalg.norm(noise[np.asarray(test)[same]], axis=1) + 1e-6
        gap = np.linalg.norm(alpha_test[same] - alpha2[same], axis=1)
        paper_r = float(np.clip(1.0 - gap / delta, 0.0, 1.0).mean())
    # Graph-structure attack on the test rows: score them with the rewired neighborhood.
    attacked_services = services.copy()
    for i in test:
        choices = catalog[catalog != services[i]]
        if len(choices):
            attacked_services[i] = choices[int(rng.integers(0, len(choices)))]
    neigh_test_adv = _neighbors(attacked_services, train, k, rng)
    _Hn, _alpha, _out, _pool, logit = forward(np.asarray(test, dtype=int), neigh_test_adv)
    p_adv = _sigmoid(logit)
    return {
        "p_test": p_test,
        "p_adv": p_adv,
        "paper_r_e": paper_r,
        "attention_l2": float(np.linalg.norm(alpha_test, axis=1).mean()),
    }


def _loss_higher(forward, batch, neigh, neigh_adv, y):
    _Hn, _alpha, out, _pool, logit = forward(batch, neigh)
    p = _sigmoid(logit)
    clean = float(-(y[batch] * np.log(np.clip(p, 1e-6, 1)) + (1 - y[batch]) * np.log(np.clip(1 - p, 1e-6, 1))).mean())
    _Hn, _alpha, out, _pool, logit = forward(batch, neigh_adv)
    p = _sigmoid(logit)
    dirty = float(-(y[batch] * np.log(np.clip(p, 1e-6, 1)) + (1 - y[batch]) * np.log(np.clip(1 - p, 1e-6, 1))).mean())
    return dirty > clean
