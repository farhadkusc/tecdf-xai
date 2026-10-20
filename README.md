# TECDF

A service-neighborhood explanation for NSL-KDD, with a numeric baseline and a service-rewire attack.

Author: **Md Farhad Kabir**. Package **0.2.4**. The research draft is [docs/PAPER.md](docs/PAPER.md). The same text is `docs/XAI_Concept_Paper.docx`. Both are generated from `benchmarks/results/evaluation.json`.

## Data

`data/raw/KDDTrain+_20Percent.txt` is the 20% NSL-KDD training split (25,192 flows), fetched 2026-10-19 from the public mirror `tuanio/nsl_kdd`. The column order is the standard 41 KDD features, the attack label, and the difficulty score. Cite Tavallaee, Bagheri, Lu, and Ghorbani, IEEE CISDA 2009, if you use the set.

## Models

Stratified 70/30 split, seed 7. Neighborhoods are sampled from the training split only.

- **Graph attention.** A flow attends over 16 other training flows that share its service. The weight of a neighbor is a softmax over the dot product of the two standardized feature vectors. The classifier sees the flow's own features, the attention-weighted neighbor features, and the attention-weighted training label.
- **Service rate.** The attack rate of the flow's service on the training split. Every training flow of that service is used, not a sample.
- **Logistic.** L2 logistic regression on ten `log1p` numeric features.
- **Rewire.** The attention classifier is scored again after every test flow is moved to a different service. The weights are not refit.

| Model | Accuracy | Precision | Recall | F1 | Average precision | R_e |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always-normal majority | 0.5339 | | | | | |
| Graph attention | 0.9624 | 0.9687 | 0.9500 | 0.9593 | 0.9852 | 0.9001 |
| Graph attention after a full service rewire | 0.6712 | 0.5960 | 0.9148 | 0.7218 | 0.9269 | |
| Service neighborhood | 0.9131 | 0.9372 | 0.8720 | 0.9034 | 0.9481 | |
| Numeric logistic | 0.9451 | 0.9507 | 0.9305 | 0.9405 | 0.9789 | 0.1177 |
| Service neighborhood after 20% rewire | 0.8265 | 0.7862 | 0.8623 | 0.8225 | 0.6439 | |

Attention entropy on the test set is 2.459. A uniform distribution over 16 neighbors has entropy 2.773. The weights are not uniform. Under N(0, 0.15) feature noise, 99.59% of test rows keep their predicted class, and R_e on those attention rows is 0.9001. The same noise on the logistic attribution gives cosine similarity 0.9703 and R_e 0.1177. Moving every test flow to a different service drops attention accuracy from 0.9624 to 0.6712. Average precision falls only from 0.9852 to 0.9269, because the flow's own features remain in the classifier.

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
