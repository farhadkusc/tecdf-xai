# Validation report

**Paper.** A Graph-Based Explainable AI Framework with Stability-Centric Evaluation for Adversarially Robust Cybersecurity Decision Systems Protecting U.S. Critical Infrastructure. Research concept paper, Md Farhad Kabir. Framework name TECDF.

**Repository.** [farhadkusc/tecdf-xai](https://github.com/farhadkusc/tecdf-xai), package 0.2.3, commit `358b6a4` (`uploaded as public`).

**Checked against.** `benchmarks/results/evaluation.json`, `src/tecdf/framework.py`, `data/raw/KDDTrain+_20Percent.txt`, README.

**Report date.** 20 October 2026.

## 1. Purpose

This report checks the concept paper against the public repository. The paper proposes a five-layer framework and a validation plan. It does not contain a filled results table. This report records the one experiment the repository does run, and sets that experiment next to the paper's definitions.

## 2. What the paper specifies

Section 4 defines a graph `G = (V, E, X)` and a graph attention network

`h_v^(k+1) = σ( W_k h_v^(k) + Σ_{u ∈ N(v)} α_uv^(k) W_k h_u^(k) )`.

Training is two-stage. Stage 1 is adversarial training in the sense of Madry et al. (2018), extended to graph edits. Stage 2 trains an explanation generator on the frozen detector, with fidelity, stability, and sparsity terms.

Two attacks are defined. A graph-structure attack edits edges and nodes inside a budget and tries to change the label. An explanation attack maximizes a divergence between explanations while the classification stays the same.

Stability, called the primary technical metric, is

`R_e(x) = 1 − D(g(x), g(x+δ)) / (||δ|| + ε_0)`.

Values near 1 are the target.

Section 5 names CICIDS2017, UNSW-NB15, Bot-IoT, phishing mail, and financial transactions. Baselines named are random forest, XGBoost, CNN/LSTM, a graph convolutional network, SHAP, LIME, and attention without a stability term. Further metrics are accuracy, precision, recall, F1, AUC, fidelity, sparsity, latency, and a human triage study.

Section 7 of this paper states that QuantumShield is under review at PLOS ONE and that DynSciGraph is under review at Machine Learning (Springer Nature). The other two concept papers, both dated April 2026, say empirical results are not yet available and place publication in a later phase. The three attachments do not agree on publication status. The repositories do not contain a submission record.

## 3. What the repository computes

Data: NSL-KDD `KDDTrain+_20Percent.txt`, 25,192 flows, from the public mirror `tuanio/nsl_kdd`, fetched 19 October 2026. Cite Tavallaee, Bagheri, Lu, and Ghorbani, IEEE CISDA 2009.

Split: stratified 70/30, seed 7. Training rows 17,634. Test rows 7,558. Test attack rate 0.4661. Neighborhood rates are computed on the training split only.

**Graph score.** The attack rate, on the training split, of the flow's `service` value. Flows that share a service are the neighborhood. There is no attention coefficient and no learned `W_k`.

**Numeric score.** L2 logistic regression on ten `log1p` features: duration, src_bytes, dst_bytes, count, srv_count, serror_rate, same_srv_rate, dst_host_count, dst_host_srv_count, dst_host_serror_rate.

**Service rewire.** Twenty percent of test flows are assigned a different service drawn from the training catalog. Labels are unchanged. The graph score is recomputed. The logistic score is not.

**Stability.** Gaussian noise `N(0, 0.15)` is added to the raw numeric features. `D` is the Euclidean distance between the logistic attribution vectors `Z ⊙ w` before and after the noise. `ε_0` in the code is `1e-6`. The result is clipped to `[0, 1]`. Cosine similarity of the same two vectors is reported beside `R_e`. The noise draw is not constrained to preserve the predicted label.

`evaluation.json` has no package version field. `src/tecdf/__init__.py` is 0.2.3.

## 4. Numerical results

| Model | Accuracy | Precision | Recall | F1 | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always-normal majority | 0.5339 | — | — | — | — |
| Service neighborhood | 0.9131 | 0.9372 | 0.8720 | 0.9034 | 0.9481 |
| Numeric logistic | 0.9451 | 0.9507 | 0.9305 | 0.9405 | 0.9789 |
| Service neighborhood after 20% rewire | 0.8265 | 0.7862 | 0.8623 | 0.8225 | 0.6439 |

Explanation cosine stability: 0.9703.

Paper formula `R_e`: 0.1177.

Standardized logistic weights: duration 0.0712, src_bytes −0.3873, dst_bytes −1.2414, count 0.7204, srv_count 0.3068, serror_rate 0.3870, same_srv_rate −0.7519, dst_host_count −0.3953, dst_host_srv_count −0.8926, dst_host_serror_rate 0.4085.

Both learned scores clear the majority accuracy of 0.5339. The logistic score is higher than the service graph on accuracy, F1, and average precision. The rewire costs 0.0866 accuracy (0.9131 → 0.8265) and 0.3042 average precision (0.9481 → 0.6439).

## 5. Reading `R_e`

The code implements the expression in Section 4.5, with `D` taken as Euclidean distance of attribution vectors. The mean on this draw is 0.1177. Under the paper's statement that stability near 1 is the target, this draw does not show a stable explanation. Cosine similarity on the same vectors is 0.9703. Cosine is a different function. It is not a substitute for `R_e` in the paper.

The noise is isotropic Gaussian noise. Section 4.5's explanation attack is a maximization of `D` subject to the detector output staying the same. That constrained attack is not what the file contains.

## 6. Claim check

| Paper claim | Repository | Result |
| --- | --- | --- |
| Graph attention network, learned `α_uv` | Service attack rate | Different model |
| Two-stage Madry training, then a frozen explanation generator | One logistic fit. No inner maximization. | Absent |
| Graph-structure attack inside a budget, aimed at a label flip | Random move of 20% of test services. Labels held fixed. Average precision falls from 0.9481 to 0.6439. | Partial |
| Explanation attack with the label held fixed | Unconstrained `N(0, 0.15)` noise on features | Different procedure |
| `R_e` near 1 as the primary result | `R_e` = 0.1177 | Formula present. Target value not met. |
| Attention, subgraph, path, and counterfactual explanations | One attribution vector, `Z ⊙ w` | Partial |
| Human oversight layer, audit log, analyst study | Not in the repository | Absent |
| CICIDS2017, UNSW-NB15, Bot-IoT, phishing, fraud graphs | NSL-KDD 20% training file only | Different data |
| Random forest, XGBoost, CNN/LSTM, GCN, SHAP, LIME | Majority class, service rate, L2 logistic | Different baselines |
| AUC, fidelity, sparsity, latency | Not in `evaluation.json` | Absent |
| Grey Harbor | Synthetic service spoof on `domain_u` inside `data/dummy/`. Separate from the NSL-KDD table. | Fixture, not a paper dataset |

## 7. Finding

On the NSL-KDD 20% file, a service-neighborhood rate reaches accuracy 0.9131 and a ten-feature logistic regression reaches 0.9451, against a majority floor of 0.5339. Rewiring the service of 20% of test flows drops the neighborhood average precision from 0.9481 to 0.6439. The paper's stability formula, applied to the logistic attribution under Gaussian noise, returns 0.1177.

The graph attention model, the adversarial training loop, the constrained explanation attack, the named public intrusion corpora, and the analyst study are not in this repository. The logistic model outscoring the service graph is a result about this file. It is not a result about the architecture in Section 4.

## 8. Source of the numbers

Regenerate with:

```bash
python3 benchmarks/run_benchmark.py
```

The split seed is 7. The input file is `data/raw/KDDTrain+_20Percent.txt`.
