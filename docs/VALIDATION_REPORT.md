# Validation report

**Paper.** TECDF concept paper, in `docs/XAI_Concept_Paper.docx`.

**Repository.** Package 0.2.3. Figures below are `benchmarks/results/evaluation.json`.

The detector is the one-layer graph attention update in `src/tecdf/gat.py`, trained with one inner step that keeps a rewired neighborhood when the loss rises. R_e is the paper's ratio, applied to the attention row when the predicted class does not change.

On the NSL-KDD 20% split, seed 7, 7,558 test flows:

| Model | Accuracy | F1 | Average precision | R_e |
| --- | ---: | ---: | ---: | ---: |
| Majority class | 0.5339 | | | |
| Graph attention | 0.9025 | 0.8887 | 0.9326 | 0.9933 |
| Graph attention after a service rewire | 0.4706 | 0.5729 | 0.4618 | |
| Service neighborhood | 0.9131 | 0.9034 | 0.9481 | |
| Numeric logistic | 0.9451 | 0.9405 | 0.9789 | 0.1177 |

The service-neighborhood row is the baseline. The logistic R_e uses the same formula on the attribution vector.
