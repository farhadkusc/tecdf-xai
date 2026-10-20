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

Layers in the concept paper that are not in this repository: stakeholder-specific views, an audit log of analyst overrides, and a human timing study. They are not filled with placeholder scores.
