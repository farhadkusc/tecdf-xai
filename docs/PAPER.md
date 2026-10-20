# TECDF

Feature attention over the service neighborhood on the NSL-KDD 20 percent training file, with the stability ratio R_e and a service-field rewire.

Md Farhad Kabir

Research draft for package 0.2.4. Every figure in this draft is taken from `benchmarks/results/evaluation.json` or `benchmarks/results/dummy_evaluation.json` in farhadkusc/tecdf-xai.

## Abstract

Each flow is a node. Its neighbors are 16 other flows from the training split that share its service. The attention weight of a neighbor is a softmax over the dot product of the two standardized feature vectors, divided by the square root of the feature dimension. An L2 logistic regression then sees three blocks: the flow's own features, the attention-weighted neighbor features, and the attention-weighted training label.

The file is `data/raw/KDDTrain+_20Percent.txt`, with 25192 flows. The split is stratified 70/30 at seed 7, giving 17634 training flows and 7558 test flows. The test attack rate is 0.4661. Predicting the majority class, which is the normal class, has accuracy 0.5339.

On the test split the attention classifier has accuracy 0.9624, precision 0.9687, recall 0.95, F1 0.9593, and average precision 0.9852. The service-rate baseline, which uses every training flow of the service rather than a sample of 16, has accuracy 0.9131 and F1 0.9034. The numeric logistic on the same ten features, with no attention, has accuracy 0.9451 and F1 0.9405.

Moving every test flow onto a different service and recomputing attention, without refitting the logistic, changes accuracy to 0.6712 and average precision to 0.9269.

R_e compares the attention row before and after feature noise drawn from N(0, 0.15), and it averages only over rows whose predicted class at threshold 0.5 does not change. The value is 0.9001. The fraction of rows that keep the class is 0.9959.

## 1. Records

The loader in `src/tecdf/nsl.py` reads the public 20 percent NSL-KDD training split. Column order is the standard 41 KDD features, then the label, then the difficulty score. A row labeled `normal` is class 0. Every other label is class 1. The service string is column 2. Ten numeric columns are transformed by log(1 + x) before any model sees them. Tavallaee, Bagheri, Lu, and Ghorbani, A detailed analysis of the KDD CUP 99 data set, IEEE CISDA 2009.

The split function walks the two classes separately. Inside each class it shuffles with `numpy` Generator seed 7 and holds out round(0.3 · class count) rows. The result file records the split string: stratified 70/30, seed 7, training neighborhoods only.

| Quantity | Value |
| --- | ---: |
| Rows | 25192 |
| Training rows | 17634 |
| Test rows | 7558 |
| Test attack rate | 0.4661 |
| Majority-class accuracy | 0.5339 |

## 2. Attention

Let z_i be the standardized log-feature vector of flow i, with mean and standard deviation computed on the training split. A coordinate whose training standard deviation is below 10^(−6) is left unscaled by replacing that standard deviation with 1. For a query i and a neighbor set N(i) of size k = 16,

score(i, j) = (z_i · z_j) / sqrt(10),

α_ij = exp(score(i, j)) / Σ_{u ∈ N(i)} exp(score(i, u)),

with the usual max-subtraction before the exponential. The pooled feature is Σ_j α_ij z_j. The pooled label is Σ_j α_ij y_j, and y_j is the training label. A flow is removed from its own neighbor pool when the pool still has another member, so the pooled label of a training flow is not a copy of its own label.

The classifier features are the concatenation of z_i, the pooled feature, and the pooled label. That vector has 10 + 10 + 1 = 21 coordinates. A second standardization, fit on the training concatenations, feeds an L2 logistic regression: 250 steps, learning rate 0.35, L2 coefficient 0.01, intercept initialized at the log-odds of the training positive rate. The decision threshold used for accuracy, precision, recall, and F1 is 0.5. Average precision uses the probability ranking and does not use that threshold.

Neighbors are drawn by `numpy` Generator seed 7. If a service has at most 16 training flows, all of them are used and the index vector is resized to length 16. Test flows never enter the neighbor pools.

Entropy of a row is −Σ_j α_ij log(α_ij + 10^(−12)). The test-set mean is 2.4591. The entropy of the uniform distribution on 16 neighbors is log(16) = 2.7726. The gap between those two numbers is the amount by which the learned weights depart from a flat average. The mean L2 norm of an attention row is 0.3189. A uniform row would have L2 norm sqrt(16 · (1/16)²) = 0.25. The reported norm is higher, which is the same fact as the entropy gap: some neighbors receive more mass than 1/16.

## 3. Service rate and numeric logistic

The service-rate score of a test flow is the fraction of training flows with the same service whose label is the attack class. A service unseen in training falls back to the global training attack rate. Every training flow of the service is used. This estimator does not sample 16 neighbors and does not use the numeric features.

The numeric logistic uses the ten log features only. It is fit for 400 steps at learning rate 0.4 with L2 coefficient 0.01, starting from a zero weight vector. Its standardized weights, in the column order of the loader, are:

| Feature | KDD column index | Standardized weight |
| --- | ---: | ---: |
| duration | 0 | 0.0712 |
| src_bytes | 4 | -0.3873 |
| dst_bytes | 5 | -1.2414 |
| count | 23 | 0.7204 |
| srv_count | 24 | 0.3068 |
| serror_rate | 25 | 0.387 |
| same_srv_rate | 28 | -0.7519 |
| dst_host_count | 31 | -0.3953 |
| dst_host_srv_count | 32 | -0.8926 |
| dst_host_serror_rate | 37 | 0.4085 |

The largest negative weights are dst_bytes at -1.2414, dst_host_srv_count at -0.8926, and same_srv_rate at -0.7519. The largest positive weights are count at 0.7204, dst_host_serror_rate at 0.4085, and serror_rate at 0.387. Duration has the smallest magnitude, 0.0712.

## 4. Stability

Two applications of the ratio are in the result file.

For attention, δ is a matrix of independent N(0, 0.15) noise added to every standardized feature, including training rows, before attention is recomputed. D is the Euclidean distance between the original attention row and the noisy attention row. The denominator is the Euclidean norm of the noise on that flow, plus 10^(−6). The ratio is

R_e = 1 − D / (||δ|| + 10^(−6)),

clipped to [0, 1], and then averaged over test rows whose class at threshold 0.5 is unchanged. The unchanged fraction is 0.9959. The mean ratio on those rows is 0.9001.

For the numeric logistic, the explanation is the coordinate-wise product of the standardized features and the weight vector. The same noise scale, 0.15, is added to the raw log features of the test rows only. Cosine similarity of the two attribution vectors, averaged over the test set, is 0.9703. The same ratio R_e on those attribution vectors is 0.1177. The two R_e numbers use the same formula and the same noise scale. They are not the same object: one is an attention row of length 16, the other is an attribution vector of length 10.

## 5. Rewires

The full rewire assigns every test flow a service drawn uniformly from the training services other than its own, using the same Generator. Neighbor pools are rebuilt. The logistic weights stay at the values fit on the original graph. Accuracy falls from 0.9624 to 0.6712. Precision falls from 0.9687 to 0.596. Recall moves from 0.95 to 0.9148. F1 moves from 0.9593 to 0.7218. Average precision moves from 0.9852 to 0.9269. The ranking degrades less than the 0.5-threshold accuracy, because the flow's own ten features remain in the classifier.

The 20 percent rewire is a different attack on a different score. It changes the service of a fraction 0.2 of test flows and recomputes the service-rate baseline only. Accuracy moves from 0.9131 to 0.8265. Precision is 0.7862, recall is 0.8623, F1 is 0.8225, and average precision is 0.6439.

## 6. Test metrics

Definitions. A predicted label is 1 when the probability is at least 0.5. Accuracy is the agreement rate. Precision is true positives over predicted positives. Recall is true positives over labeled positives. F1 is the harmonic mean. Average precision is Σ_k Precision(k) · rel(k) / N_relevant on the probability order, with a stable mergesort.

| Model | Accuracy | Precision | Recall | F1 | Average precision | R_e |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority class | 0.5339 | | | | | |
| Feature attention | 0.9624 | 0.9687 | 0.95 | 0.9593 | 0.9852 | 0.9001 |
| Feature attention after a full service rewire | 0.6712 | 0.596 | 0.9148 | 0.7218 | 0.9269 | |
| Service neighborhood | 0.9131 | 0.9372 | 0.872 | 0.9034 | 0.9481 | |
| Numeric logistic | 0.9451 | 0.9507 | 0.9305 | 0.9405 | 0.9789 | 0.1177 |
| Service neighborhood after a 20 percent rewire | 0.8265 | 0.7862 | 0.8623 | 0.8225 | 0.6439 | |

The attention classifier is ahead of the numeric logistic on accuracy (0.9624 against 0.9451), on F1 (0.9593 against 0.9405), and on average precision (0.9852 against 0.9789). It is ahead of the service rate on the same three (0.9131, 0.9034, 0.9481). The service rate remains ahead of the majority class.

## 7. Grey Harbor

The fixture is a separate table of synthetic flows under `data/dummy/`. Exercise SYN-EXERCISE-GREY-HARBOR. The service field of malicious flows in the blue cell is set to domain_u. Sites are SYN-SITE-RIVER-SWITCH, SYN-SITE-LOCK-METER. The blue cell is FED-DOE, FED-CISA, FEDLAB-INL, IBM-XFORCE, IBM-FEDERAL, RES-CMU-SEI. The out-of-scope organization is IBM-QUANTUM. Participant flows: 120. Spoofed flows: 49. Recall on that slice is 0.6471 before the spoof and 0.0 after it. The background attack rate of domain_u is 0.084. The organization file contains 106 scenario labels.

Background rates recorded with the fixture:

| Service | Background attack rate |
| --- | ---: |
| domain_u | 0.084 |
| ftp | 0.5015 |
| http | 0.8067 |
| private | 0.3537 |
| smtp | 0.6615 |
| ssh | 0.1988 |

## 8. Forward pass

The following order is the body of `fit_gat` and `evaluate_arrays`.

1. Load 25192 rows. Apply log(1 + x) to the ten numeric columns. Map `normal` to 0 and every other label to 1.
2. Stratify with seed 7 and a 0.30 test fraction. The resulting sizes are 17634 and 7558.
3. Standardize the ten features on the training rows.
4. For each flow, draw up to 16 training neighbors of the same service, excluding the flow itself when the pool allows it.
5. Compute attention, the pooled feature, and the pooled training label.
6. Fit the 21-coordinate logistic on the training concatenations.
7. Score the test concatenations. Record accuracy, precision, recall, F1, and average precision at the values in the feature-attention row.
8. Add N(0, 0.15) noise, rescore, and average R_e over the rows whose class is unchanged. The unchanged fraction is 0.9959 and the ratio is 0.9001.
9. Rewire every test service, rebuild neighbors, and rescore with the original weights. Accuracy is 0.6712.
10. Separately, fit the ten-feature logistic and the service-rate table. Apply the 20 percent service rewire to the service-rate table only. The rewired fraction in the result file is 0.2.

The attention logistic and the numeric logistic are different fits. They do not share a weight vector. Comparing their accuracies, 0.9624 and 0.9451, compares two trained classifiers on the same split, not two readouts of one model.

## 9. Software

| Path | Role |
| --- | --- |
| src/tecdf/nsl.py | Loader, column index, log1p |
| src/tecdf/gat.py | Neighbor sample, attention, R_e, full rewire |
| src/tecdf/framework.py | Split, service rate, numeric logistic, 20 percent rewire |
| benchmarks/run_benchmark.py | Writes evaluation.json |
| lab/njit/run_lab.py | Service-label permutation job for the NJIT Wulver cluster |

The ten feature names in the result file, in order, are duration, src_bytes, dst_bytes, count, srv_count, serror_rate, same_srv_rate, dst_host_count, dst_host_srv_count, dst_host_serror_rate.

| duration | src_bytes | dst_bytes | count | srv_count | serror_rate | same_srv_rate | dst_host_count | dst_host_srv_count | dst_host_serror_rate |
| --- |
| 0.0712 | -0.3873 | -1.2414 | 0.7204 | 0.3068 | 0.387 | -0.7519 | -0.3953 | -0.8926 | 0.4085 |

## 10. Reproduce

python3 -m unittest tests/test_framework.py tests/test_dummy.py tests/test_lab_smoke.py

python3 benchmarks/run_benchmark.py

python3 benchmarks/run_dummy.py

Code path: `fit_gat` in `src/tecdf/gat.py`, called from `evaluate_arrays` in `src/tecdf/framework.py`, with seed 7 and k = 16.
