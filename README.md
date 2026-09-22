# EvoGC

M. Ožbot and I. Škrjanc, "Evolving Beyond Gaussian Prototypes with Multi-Scale Graph Consensus on Microclusters", IEEE EAIS 2026, Pisa.

![Fig. 1, the pipeline](figures/fig1_pipeline.png)

![Fig. 2, the steps on Jain](figures/fig2_steps.png)

```python
model = EvoGC(K=2, random_state=0).fit(X)
labels = model.pooled_row["labels"]
```
