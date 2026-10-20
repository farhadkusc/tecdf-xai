# TECDF

Trustworthy Explainable Cybersecurity Decision Framework. Graph attention detection, two-stage training, and an explanation-stability metric for critical-infrastructure decision systems.

Author: **Md Farhad Kabir**

## Package

| Path | Role |
| --- | --- |
| `src/tecdf/framework.py` | Synthetic flow graphs, attention pool, edge-flip attack, Stage-1 trainer, \(R_e\) stability |
| `benchmarks/run_benchmark.py` | Three seeds, writes `benchmarks/results/stability.json` |
| `docs/TECHNICAL_REPORT.md` | Metric definitions and limits |
| `docs/ARCHITECTURE.md` | Five layers and what this package implements |
| `tests/test_framework.py` | Stability bounds and a learning check |

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest tests/test_framework.py
python3 benchmarks/run_benchmark.py
```

## Metrics

- **Accuracy** on held-out synthetic graphs.
- **Accuracy under edge attack.** Up to 3 added edges. This is the graph-structure threat in the concept paper, at a fixed budget.
- **Stability \(R_e\).** Attention L2 change under an L2 feature perturbation of size 0.15.
- **Fidelity.** Fraction of test graphs where deleting the highest-attention node moves the probability at least as much as deleting the lowest-attention node.
- **Sparsity.** Attention mass on the top 3 nodes.

## Honest scope

No CICIDS2017, UNSW-NB15, or analyst study is in this repository. The five-layer architecture (detection, explanation, human oversight, audit, adversarial resilience) is described in `docs/ARCHITECTURE.md`. The code implements detection, explanation, and the resilience metric. Oversight and audit are specified, not staffed.

## License

MIT. See `LICENSE`.
