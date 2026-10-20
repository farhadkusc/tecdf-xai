# Validation report

**Paper.** TECDF concept paper, `docs/XAI_Concept_Paper.docx`.

**Repository.** Package 0.2.4. Figures are `benchmarks/results/evaluation.json`.

Attention is a softmax over feature dot products on 16 other training flows of the same service. The classifier uses the flow, the pooled features, and the pooled training label. R_e uses N(0, 0.15) noise and only rows whose predicted class does not change.

| Model | Accuracy | F1 | Average precision | R_e |
| --- | ---: | ---: | ---: | ---: |
| Majority class | 0.5339 | | | |
| Graph attention | 0.9624 | 0.9593 | 0.9852 | 0.9001 |
| Graph attention after a full service rewire | 0.6712 | 0.7218 | 0.9269 | |
| Service neighborhood | 0.9131 | 0.9034 | 0.9481 | |
| Numeric logistic | 0.9451 | 0.9405 | 0.9789 | 0.1177 |

Test attention entropy is 2.459. The uniform entropy over 16 neighbors is 2.773. The fraction of rows whose class survives the noise is 0.9959.
