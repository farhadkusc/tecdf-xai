#!/usr/bin/env python3
"""Stage-1/stage-2 evaluation on synthetic flow graphs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tecdf.framework import evaluate  # noqa: E402


def main() -> None:
    seeds = (7, 11, 19)
    rows = []
    for seed in seeds:
        ev = evaluate(seed)
        rows.append({
            "seed": seed,
            "n_test": ev.n,
            "accuracy": round(ev.accuracy, 4),
            "accuracy_under_edge_attack": round(ev.accuracy_under_edge_attack, 4),
            "mean_stability_R_e": round(ev.mean_stability, 4),
            "fidelity_top_beats_bottom": round(ev.mean_fidelity, 4),
            "sparsity_top3_mass": round(ev.mean_sparsity_top3, 4),
        })
    payload = {
        "disclaimer": (
            "Synthetic flow graphs with a planted dense core. Not CICIDS2017, "
            "UNSW-NB15, or a human analyst study."
        ),
        "protocol": "50/50 split, edge-flip attack budget 3, feature noise L2=0.15",
        "runs": rows,
    }
    out = ROOT / "benchmarks" / "results" / "stability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{'seed':>6} {'acc':>8} {'acc@atk':>8} {'R_e':>8} {'fidelity':>9} {'top3':>8}")
    for r in rows:
        print(
            f"{r['seed']:6d} {r['accuracy']:8.3f} {r['accuracy_under_edge_attack']:8.3f} "
            f"{r['mean_stability_R_e']:8.3f} {r['fidelity_top_beats_bottom']:9.3f} "
            f"{r['sparsity_top3_mass']:8.3f}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
