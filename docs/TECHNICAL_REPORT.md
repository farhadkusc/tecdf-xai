# TECDF technical report

**Package version:** 0.1.0  
**Author:** Md Farhad Kabir

## Detector

Node state is a mix of its own features and the mean of its out-neighbors. A learned query produces a softmax attention distribution over nodes. The graph vector is the attention-weighted sum of node states. A linear head maps that vector to a malicious/benign probability.

Stage 1 alternates clean graphs and 2-edge insertions, matching the inner-max / outer-min shape of adversarial training at a scale that runs without a GPU. The query is updated by a short accept/reject search so the explanation is not an untrained uniform vector. Stage 2 does not update the detector. Fidelity and stability are measurements, not a second loss that is back-propagated. That split is the two-stage strategy in the concept paper.

## Stability

\[
R_e(x) = 1 - \frac{\|g(x) - g(x+\delta)\|_2}{\|\delta\|_2 + \epsilon_0}
\]

with \(\epsilon_0 = 10^{-6}\) and \(\|\delta\|_2 = 0.15\). Values are clipped to \([0, 1]\).

## Data

Each graph has 12 nodes. Malicious graphs contain a 4-node clique and a shift on feature 0 of those nodes. Benign graphs are Erdős–Rényi at edge probability 0.18. The test above 0.5 accuracy is whether the head learned that core, not a claim about network intrusion detection.

## Not included

SHAP, LIME, human triage timing, and the public intrusion corpora named in the concept paper. Add those in a later evaluation harness; keep this file's JSON labeled synthetic.
