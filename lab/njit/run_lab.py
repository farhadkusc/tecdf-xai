#!/usr/bin/env python3
"""Seed and rewire-fraction sweep, plus a label-permutation null, on NSL-KDD 20%.

Neighborhood rates and the logistic fit are recomputed for every seed.
The permutation null shuffles labels and rebuilds the service rates.
Smoke is the short check.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tecdf import __version__  # noqa: E402
from tecdf.framework import _service_rate, _split, evaluate_arrays  # noqa: E402
from tecdf.nsl import load  # noqa: E402

LAB_SEEDS = range(48)
LAB_FRACS = (0.05, 0.10, 0.20, 0.40, 0.60)
LAB_PERMUTATIONS = 400
SMOKE_SEEDS = (0, 1)
SMOKE_FRACS = (0.20,)
SMOKE_PERMUTATIONS = 20

_DATA = None


def _load():
    global _DATA
    if _DATA is None:
        _DATA = load()
    return _DATA


def _worker(job: tuple) -> dict:
    x, y, services = _load()
    seed, frac = job
    out = evaluate_arrays(x, y, services, seed=int(seed), attack_frac=float(frac))
    graph = out["graph_service_neighborhood"]
    logistic = out["numeric_logistic"]
    attacked = out["graph_after_service_rewire"]
    return {
        "seed": int(seed),
        "attack_frac": float(frac),
        "graph_accuracy": graph["accuracy"],
        "graph_f1": graph["f1"],
        "graph_ap": graph["average_precision"],
        "logistic_accuracy": logistic["accuracy"],
        "logistic_f1": logistic["f1"],
        "logistic_ap": logistic["average_precision"],
        "rewire_accuracy": attacked["accuracy"],
        "rewire_ap": attacked["average_precision"],
        "majority_class_accuracy": out["majority_class_accuracy"],
    }


def _permutation_null(n_perm: int, seed: int = 7) -> dict:
    x, y, services = _load()
    del x
    train, test = _split(len(y), y, seed=seed)
    rate = _service_rate(services[train], y[train])
    base = float(y[train].mean())
    pred = np.array([1 if rate.get(services[i], base) >= 0.5 else 0 for i in test])
    observed = float((pred == y[test]).mean())
    rng = np.random.default_rng(seed + 99)
    null = np.empty(n_perm, dtype=float)
    y_work = y.copy()
    for i in range(n_perm):
        rng.shuffle(y_work)
        rate_i = _service_rate(services[train], y_work[train])
        base_i = float(y_work[train].mean())
        pred_i = np.array([1 if rate_i.get(services[j], base_i) >= 0.5 else 0 for j in test])
        null[i] = float((pred_i == y_work[test]).mean())
    return {
        "split_seed": seed,
        "n_permutations": n_perm,
        "observed_graph_accuracy": round(observed, 4),
        "null_mean": round(float(null.mean()), 4),
        "null_max": round(float(null.max()), 4),
        "fraction_null_ge_observed": round(float(np.mean(null >= observed)), 4),
    }


def _mean(rows: list[dict], key: str) -> float:
    return round(sum(r[key] for r in rows) / len(rows), 4)


def _summary(rows: list[dict]) -> list[dict]:
    fracs = sorted({r["attack_frac"] for r in rows})
    out = []
    for frac in fracs:
        items = [r for r in rows if r["attack_frac"] == frac]
        out.append(
            {
                "attack_frac": frac,
                "n_seeds": len(items),
                "graph_accuracy_mean": _mean(items, "graph_accuracy"),
                "logistic_accuracy_mean": _mean(items, "logistic_accuracy"),
                "rewire_accuracy_mean": _mean(items, "rewire_accuracy"),
                "rewire_ap_mean": _mean(items, "rewire_ap"),
                "accuracy_drop_mean": round(_mean(items, "graph_accuracy") - _mean(items, "rewire_accuracy"), 4),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="NJIT Wulver sweep for TECDF")
    parser.add_argument("--profile", choices=("smoke", "lab"), default="smoke")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    if args.profile == "smoke":
        seeds, fracs, n_perm = SMOKE_SEEDS, SMOKE_FRACS, SMOKE_PERMUTATIONS
    else:
        seeds, fracs, n_perm = LAB_SEEDS, LAB_FRACS, LAB_PERMUTATIONS
    jobs = [(s, f) for s in seeds for f in fracs]
    started = time.perf_counter()
    workers = max(1, min(args.workers, len(jobs)))
    if workers == 1:
        rows = [_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_worker, jobs))
    null = _permutation_null(n_perm)
    elapsed = round(time.perf_counter() - started, 3)
    report = {
        "package": "tecdf-xai",
        "package_version": __version__,
        "profile": args.profile,
        "host": socket.gethostname(),
        "ran_on_wulver": False,
        "purpose": "Seed and service-rewire sweep on the NSL-KDD 20% file, with a label-permutation null for the service graph.",
        "n_jobs": len(rows),
        "workers": workers,
        "elapsed_sec": elapsed,
        "summary": _summary(rows),
        "permutation_null": null,
        "jobs": rows,
    }
    out_dir = ROOT / "lab" / "njit" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "smoke.json" if args.profile == "smoke" else "lab.json"
    path = out_dir / name
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"jobs {len(rows)} perms {n_perm} elapsed {elapsed}s -> {path}")


if __name__ == "__main__":
    main()
