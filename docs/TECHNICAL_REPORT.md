# Technical report

**Version 0.2.2.** Author: Md Farhad Kabir.

## Graph

Each flow is a node. Training flows that share `service` are its neighborhood. The graph score is the fraction of that neighborhood labeled attack. Services unseen in training get the training base rate.

The numeric model uses, in KDD order: duration, src_bytes, dst_bytes, count, srv_count, serror_rate, same_srv_rate, dst_host_count, dst_host_srv_count, dst_host_serror_rate. Inputs are `log1p`. Coefficients in the results file are on the standardized scale.

## Attack

A graph-structure attack in this dataset is a lie about service, because service is the edge relation. Rewiring 20% of test nodes to a uniformly chosen other known service drops accuracy from 0.913 to 0.827 and average precision from 0.948 to 0.644. Accuracy understates the attack: many rewired flows are still on the correct side of 0.5, but the ranking is much worse. The logistic model does not use service, so this particular attack is not aimed at it. The majority-class accuracy on the same test split is 0.534 (always normal).

## Explanation

The attribution vector is the standardized numeric features times the logistic weights. Cosine stability is the mean cosine between that vector and the same vector after isotropic Gaussian noise of scale 0.15. \(R_e\) uses

\[
R_e = 1 - \frac{\|a - a'\|_2}{\|\delta\|_2 + \epsilon}
\]

clipped to [0, 1]. Mean \(R_e\) is 0.118 because the weight vector amplifies some coordinates and the ratio saturates. It is reported so the formula can be compared with the cosine on the same noise draw.

## Split

Seed 7, 30% test, stratified on the binary label (normal versus any attack). There is no second external test file in this package; the official NSL-KDD test split is a known distribution shift and is not claimed here.

## Synthetic fixtures

`data/dummy/flows.csv` is a planted service neighborhood, twenty rows per scenario label. The labels are the same federal, IBM, and research-center registry used by the other packages. `evaluate_arrays` scores it. The NSL-KDD numbers in `evaluation.json` are not recomputed from this file.

Grey Harbor scores the blue-cell slice twice. The second pass rewrites the service of malicious rows to `domain_u`. Neighborhood rates are computed on the other organizations only.
