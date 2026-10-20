# Architecture

```mermaid
flowchart LR
  F[NSL-KDD flow] --> S[Service neighborhood<br/>training attack rate]
  F --> N[Ten numeric features]
  N --> L[L2 logistic]
  S --> E[Graph score and rewire attack]
  L --> A[Attribution vector]
  A --> R[Cosine stability and paper R_e]
```

Attention is a softmax over feature dot products inside the service neighborhood. The classifier uses the flow, the pooled neighbor features, and the pooled training label. The service rate and the numeric logistic are the baselines.
