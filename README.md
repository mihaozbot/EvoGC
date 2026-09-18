# EvoGC

Evolving clustering with graph consensus on microclusters. Code for

M. Ožbot and I. Škrjanc, "Evolving Beyond Gaussian Prototypes with Multi-Scale Graph Consensus on Microclusters", IEEE EAIS 2026, Pisa.

![Fig. 1, the pipeline](figures/fig1_pipeline.png)

An evolving compressor turns the data stream into microclusters in one pass. Shared-nearest-neighbour graphs are built on the microclusters at several scales, density peaks and a greedy merge give a partition at each scale, and a spectral consensus over all scales and repetitions gives the final clustering. The only parameter is the number of clusters K.

![Fig. 2, the steps on Jain](figures/fig2_steps.png)

Figures from the paper, © IEEE.

## Use

```python
from evogc import EvoGC

model = EvoGC(K=2, random_state=0).fit(X)
labels = model.pooled_row["labels"]
```

K is the number of clusters and is the only thing to set; the benchmarks in the paper give every method the true K. `n_comp` sets the number of compression repetitions (10 by default). If `y` is passed to `fit`, ARI, NMI and accuracy are printed.

`python evogc.py` runs the method on the benchmark sets used in the paper (Iris, Wine, MNIST and the SIPU shape sets, downloaded to `sipu_cache/` on first use).

## Comparison

`python experiments.py` runs the nine baselines from the paper on the same sets and writes `results/baseline_results.csv`. It needs `hdbscan` and `river` on top of the requirements. Mean ARI over the 17 benchmark sets, Table I of the paper:

| | EvoGC | Leiden | DPA | HDBSCAN | Ward | BIRCH | K-Means | SubKMeans | FINCH | Spectral |
|---|---|---|---|---|---|---|---|---|---|---|
| mean ARI | 0.71 | 0.62 | 0.54 | 0.54 | 0.50 | 0.46 | 0.46 | 0.46 | 0.41 | 0.39 |
| mean rank | 2.7 | 3.0 | 4.9 | 6.1 | 5.3 | 6.1 | 6.4 | 6.9 | 6.5 | 6.4 |

## Install

```
pip install -r requirements.txt
```

numba and joblib are optional; without them the same code runs, only slower.

## Files

`evogc.py` the method, `evogc_data.py` the dataset loaders, `experiments.py` the baseline comparison, `figures/` Fig. 1 to 4 from the paper.
