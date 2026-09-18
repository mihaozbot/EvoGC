"""Baseline comparison on the benchmark datasets, writes results/baseline_results.csv."""
from __future__ import annotations
import io, time, warnings, csv
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
import hdbscan
from river import stream as river_stream
from river.cluster import STREAMKMeans, CluStream, DenStream, DBSTREAM

from evogc_data import DATASET_REGISTRY, load_dataset
from evogc import EvoGC, clustering_accuracy

OUT = Path(__file__).parent / "results"
SEED = 0
BIG_N = 8000
STREAM_MAX_N = 8000

DATASETS = ["mnist", "iris", "wine", "sipu_spiral", "sipu_jain", "sipu_D31",
            "sipu_R15", "sipu_a1", "sipu_s3", "sipu_s4", "sipu_aggregation",
            "sipu_compound", "sipu_flame", "sipu_path_based", "sipu_unbalance"]


def metrics(y, pred):
    return float(ARI(y, pred)), float(NMI(y, pred)), float(clustering_accuracy(y, pred)), int(len(np.unique(pred)))


def run_kmeans(X, K, seed):
    return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(X)


def run_gmm(X, K, seed):
    return GaussianMixture(n_components=K, n_init=3, random_state=seed).fit_predict(X)


def run_spectral(X, K, seed):
    return SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                              n_neighbors=10, random_state=seed, assign_labels="kmeans").fit_predict(X)


def run_ward(X, K, seed):
    return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)


def run_birch(X, K, seed):
    return Birch(n_clusters=K).fit_predict(X)


def run_hdbscan(X, K, seed):
    mcs = max(5, X.shape[0] // 100)
    return hdbscan.HDBSCAN(min_cluster_size=mcs).fit_predict(X)


def _stream_fit_predict(model, X):
    for xi, _ in river_stream.iter_array(X):
        model.learn_one(xi)
    preds = []
    for xi, _ in river_stream.iter_array(X):
        p = model.predict_one(xi)
        preds.append(-1 if p is None else int(p))
    return np.asarray(preds)


def run_streamkmeans(X, K, seed):
    return _stream_fit_predict(STREAMKMeans(n_clusters=K, halflife=0.5, seed=seed), X)


def run_clustream(X, K, seed):
    return _stream_fit_predict(CluStream(n_macro_clusters=K, seed=seed), X)


def run_denstream(X, K, seed):
    return _stream_fit_predict(DenStream(decaying_factor=0.25, beta=0.75, mu=2, epsilon=0.5), X)


def run_dbstream(X, K, seed):
    return _stream_fit_predict(DBSTREAM(clustering_threshold=1.0, fading_factor=0.05), X)


def run_evogc(X, K, seed, y):
    m = EvoGC(K=K, random_state=seed)
    with redirect_stdout(io.StringIO()):
        m.fit(X, y=y, name="ds")
    return m.pooled_row["labels"]

METHODS = [
    ("EvoGC",        run_evogc,        True,  "proposed",  True,  False, True),
    ("KMeans",       run_kmeans,       True,  "gaussian",  True,  False, True),
    ("GMM",          run_gmm,          True,  "gaussian",  True,  False, True),
    ("Spectral",     run_spectral,     True,  "batch",     False, False, True),
    ("Ward",         run_ward,         True,  "batch",     False, False, False),
    ("Birch",        run_birch,        True,  "batch",     True,  False, False),
    ("HDBSCAN",      run_hdbscan,      False, "density",   False, False, False),
    ("STREAMKMeans", run_streamkmeans, True,  "evolving",  True,  True,  True),
    ("CluStream",    run_clustream,    True,  "evolving",  True,  True,  True),
    ("DenStream",    run_denstream,    False, "evolving",  True,  True,  False),
    ("DBSTREAM",     run_dbstream,     False, "evolving",  True,  True,  False),
]
SEEDS = [0, 1, 2]


def main():
    rows = []
    for ds in DATASETS:
        cfg = DATASET_REGISTRY[ds]
        with redirect_stdout(io.StringIO()):
            X, y = load_dataset(cfg, random_state=SEED)
        n = len(X)
        print(f"\n{cfg.name}  n={n}  d={X.shape[1]}  K={cfg.true_k}")
        print(f"{'method':<14}{'family':<10}{'ARI':>16}{'NMI':>8}{'ACC':>8}{'K':>5}{'sec':>8}")
        for name, fn, needs_K, fam, scalable, is_stream, stochastic in METHODS:
            if n > BIG_N and not scalable:
                print(f"{name:<14}{fam:<10}{'  skipped (n>'+str(BIG_N)+', O(n^2+))':>32}")
                continue
            if is_stream and n > STREAM_MAX_N:
                print(f"{name:<14}{fam:<10}{'  skipped (stream loop, n>'+str(STREAM_MAX_N)+')':>32}")
                continue
            seeds = SEEDS if stochastic else [0]
            aris, nmis, accs, ks, secs, err = [], [], [], [], [], None
            for s in seeds:
                try:
                    t0 = time.time()
                    pred = fn(X, cfg.true_k, s, y) if name == "EvoGC" else fn(X, cfg.true_k, s)
                    secs.append(time.time() - t0)
                    a, nm, ac, kf = metrics(y, pred)
                    aris.append(a)
                    nmis.append(nm)
                    accs.append(ac)
                    ks.append(kf)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"[:40]
                    break
            if err:
                print(f"{name:<14}{fam:<10}  failed: {err}")
                rows.append([cfg.name, name, fam, "nan", "nan", "nan", "nan", "nan", err])
                continue
            am, asd = np.mean(aris), np.std(aris)
            disp = f"{am:.3f}" + (f"±{asd:.3f}" if len(seeds) > 1 else "")
            print(f"{name:<14}{fam:<10}{disp:>16}{np.mean(nmis):>8.3f}{np.mean(accs):>8.3f}"
                  f"{int(np.median(ks)):>5}{np.mean(secs):>8.2f}")
            rows.append([cfg.name, name, fam, f"{am:.4f}", f"{asd:.4f}",
                         f"{np.mean(nmis):.4f}", f"{np.mean(accs):.4f}",
                         int(np.median(ks)), f"{np.mean(secs):.3f}"])

    with open(OUT / "baseline_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "family", "ari_mean", "ari_std", "nmi", "acc", "K_found", "seconds"])
        w.writerows(rows)

    print("\nmean ARI over the datasets each method ran on")
    by_method = {}
    for r in rows:
        if r[3] == "nan":
            continue
        by_method.setdefault(r[1], []).append(float(r[3]))
    for name, *_ in METHODS:
        if name in by_method:
            v = by_method[name]
            print(f"  {name:<14} mean_ARI={np.mean(v):.3f}  (over {len(v)} datasets)")
    print(f"\nwritten to {OUT / 'baseline_results.csv'}")


if __name__ == "__main__":
    main()
