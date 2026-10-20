# TECDF

Feature attention over the service neighborhood on the NSL-KDD 20 percent training split, with the stability ratio R_e and a service rewire.

Md Farhad Kabir

Research draft for package 0.2.4, aligned with `benchmarks/results/evaluation.json` and `benchmarks/results/dummy_evaluation.json`. Repository farhadkusc/tecdf-xai.

## Abstract

Each flow attends over 16 other training flows that share its service. The weight of a neighbor is a softmax over the dot product of the two standardized feature vectors, divided by the square root of the feature count. The classifier is an L2 logistic regression on the flow's own features, the attention-weighted neighbor features, and the attention-weighted training label. The split is stratified 70/30 with seed 7. On 7558 test flows the attention classifier has accuracy 0.9624, F1 0.9593, and average precision 0.9852. Moving every test flow to a different service changes accuracy to 0.6712 and average precision to 0.9269.

R_e is 1 − D(attention, attention after noise) / (||δ|| + 10^(−6)), with δ drawn from N(0, 0.15), and it is averaged only over rows whose predicted class does not change. That value is 0.9001. The fraction of rows that keep the class is 0.9959.

## 1. Data

`data/raw/KDDTrain+_20Percent.txt` has 25192 flows. Training rows 17634. Test rows 7558. Test attack rate 0.4661. Majority-class accuracy 0.5339. The file is the 20 percent NSL-KDD training split. Tavallaee, Bagheri, Lu, and Ghorbani, IEEE CISDA 2009. The result file records the split as: stratified 70/30, seed 7, training neighborhoods only.

## 2. Attention

Neighbors are sampled from the training split only, and a flow is not its own neighbor. Query and key are the standardized numeric features. The ten features are duration, src_bytes, dst_bytes, count, srv_count, serror_rate, same_srv_rate, dst_host_count, dst_host_srv_count, dst_host_serror_rate. Attention entropy on the test set is 2.4591. A uniform distribution over 16 neighbors has entropy 2.7726. Mean L2 norm of the attention row is 0.3189.

The service-rate baseline scores a test flow by the attack rate of its service on the full training split. The numeric logistic uses the same ten features after `log1p`, standardized on the training split, with no attention. Its standardized weights are below.

| duration | src_bytes | dst_bytes | count | srv_count | serror_rate | same_srv_rate | dst_host_count | dst_host_srv_count | dst_host_serror_rate |
| --- |
| 0.0712 | -0.3873 | -1.2414 | 0.7204 | 0.3068 | 0.387 | -0.7519 | -0.3953 | -0.8926 | 0.4085 |

## 3. Stability and rewire

Feature noise for the attention R_e is N(0, 0.15), the same scale used for the logistic attribution. Logistic attribution R_e on that draw is 0.1177. Cosine similarity of the logistic attribution is 0.9703.

The full rewire assigns every test flow a different service and recomputes attention and the pooled label. The logistic weights are not refit. A second rewire moves 20 percent of test flows and recomputes only the service-rate score.

## 4. Results

| Model | Accuracy | Precision | Recall | F1 | Average precision | R_e |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority class | 0.5339 | | | | | |
| Feature attention | 0.9624 | 0.9687 | 0.95 | 0.9593 | 0.9852 | 0.9001 |
| Feature attention after a full service rewire | 0.6712 | 0.596 | 0.9148 | 0.7218 | 0.9269 | |
| Service neighborhood | 0.9131 | 0.9372 | 0.872 | 0.9034 | 0.9481 | |
| Numeric logistic | 0.9451 | 0.9507 | 0.9305 | 0.9405 | 0.9789 | 0.1177 |
| Service neighborhood after a 20 percent rewire | 0.8265 | 0.7862 | 0.8623 | 0.8225 | 0.6439 | |

The 20 percent rewire fraction recorded in the result file is 0.2.

## 5. Grey Harbor

Exercise SYN-EXERCISE-GREY-HARBOR. The service field of malicious flows in the blue cell is set to domain_u. Participant flows 120. Spoofed flows 49. Recall on that slice is 0.6471 before the spoof and 0.0 after it. The background rate of domain_u is 0.084. The organization file has 106 scenario labels.

## 6. Reproduce

python3 benchmarks/run_benchmark.py

python3 benchmarks/run_dummy.py

Code: `src/tecdf/gat.py`, function `fit_gat`, called from `evaluate_arrays` in `src/tecdf/framework.py` with seed 7 and k = 16.
