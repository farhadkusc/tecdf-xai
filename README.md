# TECDF

A service-neighborhood explanation for NSL-KDD, with a numeric baseline and a service-rewire attack.

Author: **Md Farhad Kabir**. Package **0.2.3**.

## Data

`data/raw/KDDTrain+_20Percent.txt` is the 20% NSL-KDD training split (25,192 flows), fetched 2026-10-19 from the public mirror `tuanio/nsl_kdd`. The column order is the standard 41 KDD features, the attack label, and the difficulty score. Cite Tavallaee, Bagheri, Lu, and Ghorbani, IEEE CISDA 2009, if you use the set.

## Models

Stratified 70/30 split, seed 7. Neighborhoods are sampled from the training split only.

- **Graph attention.** One layer. A flow attends over eight training flows that share its service. The update is the concept-paper sum of attention-weighted neighbor states, and the neighbor label is included in that sum. One training step keeps a rewired neighborhood when it raises the loss.
- **Service rate.** The attack rate of the flow's service on the training split.
- **Logistic.** L2 logistic regression on ten `log1p` numeric features.
- **Rewire.** The attention model is scored again after every test flow is moved to a different service.

| Model | Accuracy | F1 | Average precision | R_e |
| --- | ---: | ---: | ---: | ---: |
| Always-normal majority | 0.5339 | | | |
| Graph attention | 0.9025 | 0.8887 | 0.9326 | 0.9933 |
| Graph attention after service rewire | 0.4706 | 0.5729 | 0.4618 | |
| Service neighborhood | 0.9131 | 0.9034 | 0.9481 | |
| Numeric logistic | 0.9451 | 0.9405 | 0.9789 | 0.1177 |
| Service neighborhood after 20% rewire | 0.8265 | 0.8225 | 0.6439 | |

R_e is the concept-paper ratio. On the attention rows, under feature noise that leaves the predicted class unchanged, it is 0.9933. On the logistic attribution vector under N(0, 0.15) noise it is 0.1177. Cosine similarity of that attribution is 0.9703.

## Reproduce

```bash
python3 -m unittest tests/test_framework.py
python3 benchmarks/run_benchmark.py
python3 benchmarks/plot_results.py
```

The chart is [docs/figures/results.svg](docs/figures/results.svg).

## Synthetic fixtures

`data/dummy/flows.csv` has twenty synthetic flows for each organization in the same registry the other two packages use: cabinet departments, DOE laboratories, federal components, IBM units, and research centers.

```bash
python3 -m unittest tests/test_dummy.py
python3 benchmarks/run_dummy.py
```

`benchmarks/results/dummy_evaluation.json` is the fixture check.

Grey Harbor (`SYN-EXERCISE-GREY-HARBOR`) spoofs the service field of malicious flows in the blue cell to `domain_u`. Recall on that slice falls.

## NJIT Wulver

`lab/njit/` refits the NSL-KDD models across 48 seeds and five rewire fractions, then runs a 400-draw label permutation. Submit with `sbatch lab/njit/wulver.sbatch`. Instructions are in [lab/njit/README.md](lab/njit/README.md).
