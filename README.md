# TECDF

A service-neighborhood explanation for NSL-KDD, with a numeric baseline and a service-rewire attack.

Author: **Md Farhad Kabir**. Package **0.2.3**.

## Data

`data/raw/KDDTrain+_20Percent.txt` is the 20% NSL-KDD training split (25,192 flows), fetched 2026-10-19 from the public mirror `tuanio/nsl_kdd`. The column order is the standard 41 KDD features, the attack label, and the difficulty score. Cite Tavallaee, Bagheri, Lu, and Ghorbani, IEEE CISDA 2009, if you use the set. This is not CICIDS2017, UNSW-NB15, or an analyst study.

## Models

Stratified 70/30 split, seed 7. Neighborhood rates are computed on the training split only.

- **Graph.** A flow's score is the attack rate of its service among training flows. That is a one-line neighborhood aggregate, which is the graph the service field actually induces.
- **Logistic.** L2 logistic regression on ten `log1p` numeric features (duration, bytes, count, error rates, host counts). Weights are in the results JSON.
- **Attack.** Twenty percent of test flows are moved to a different service. Their graph score changes because their neighborhood changes. Labels do not.

| Model | Accuracy | F1 | Average precision |
| --- | ---: | ---: | ---: |
| Always-normal majority | 0.534 | — | — |
| Service neighborhood | 0.913 | 0.903 | 0.948 |
| Numeric logistic | 0.945 | 0.941 | 0.979 |
| Neighborhood after 20% service rewire | 0.827 | 0.823 | 0.644 |

Always predicting normal, the majority class, is 0.534 accuracy on this stratified test split. Both models clear that. The numeric model wins on accuracy, F1, and average precision. A partial service rewire costs about 8.6 accuracy points and, more sharply, drops average precision from 0.948 to 0.644. That ranking collapse is the graph-structure attack. Explanation cosine stability under N(0, 0.15) noise on the numeric features is 0.970. The paper's \(R_e\), using the raw L2 gap of attribution vectors over the noise norm, is 0.118 on the same draw. Both are reported. The cosine stays informative when attribution weights are larger than the noise; \(R_e\) is the formula in the concept paper and it saturates here.

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
