"""Dataset loaders: sklearn builtins, OpenML, and the SIPU benchmarks (cached in sipu_cache/)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import urllib.request
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

SIPU_BASE_URL = "https://cs.uef.fi/sipu/datasets"
SIPU_CACHE_DIR = Path(__file__).parent / "sipu_cache"


@dataclass
class DatasetConfig:
    key: str
    name: str
    true_k: int | None
    loader: str
    loader_kwargs: dict = field(default_factory=dict)
    pca_dims: int | None = None
    scale: str | None = "standard"

DATASET_REGISTRY: dict[str, DatasetConfig] = {

    "iris": DatasetConfig("iris", "Iris", true_k=3,
        loader="sklearn_builtin", loader_kwargs={"fn": "load_iris"}),
    "wine": DatasetConfig("wine", "Wine", true_k=3,
        loader="sklearn_builtin", loader_kwargs={"fn": "load_wine"}),
    "breast_cancer": DatasetConfig("breast_cancer", "Breast Cancer", true_k=2,
        loader="sklearn_builtin", loader_kwargs={"fn": "load_breast_cancer"}),
    "digits": DatasetConfig("digits", "Digits", true_k=10,
        loader="sklearn_builtin", loader_kwargs={"fn": "load_digits"}),

    "pendigits": DatasetConfig("pendigits", "Pendigits", true_k=10,
        loader="openml", loader_kwargs={"name": "pendigits", "version": 1}),
    "optdigits": DatasetConfig("optdigits", "Optdigits", true_k=10,
        loader="openml", loader_kwargs={"name": "optdigits", "version": 1}),
    "mnist": DatasetConfig("mnist", "MNIST", true_k=10,
        loader="openml", loader_kwargs={"name": "mnist_784", "version": 1},
        scale="center", pca_dims=50),
    "usps": DatasetConfig("usps", "USPS", true_k=10,
        loader="openml", loader_kwargs={"name": "usps", "version": 2}, pca_dims=50),
    "fashion_mnist": DatasetConfig("fashion_mnist", "FashionMNIST", true_k=10,
        loader="openml", loader_kwargs={"name": "Fashion-MNIST", "version": 1},
        scale="center", pca_dims=50),
    "kmnist": DatasetConfig("kmnist", "KMNIST", true_k=10,
        loader="openml", loader_kwargs={"name": "Kuzushiji-MNIST", "version": 1},
        scale="center", pca_dims=50),
    "mfeat_karhunen": DatasetConfig("mfeat_karhunen", "MfeatKarhunen", true_k=10,
        loader="openml", loader_kwargs={"name": "mfeat-karhunen", "version": 1}),
    "banknote": DatasetConfig("banknote", "Banknote", true_k=2,
        loader="openml", loader_kwargs={"name": "banknote-authentication", "version": 1},
        scale=None),
    "magic": DatasetConfig("magic", "Magic", true_k=2,
        loader="openml", loader_kwargs={"name": "MagicTelescope", "version": 1}),

    "sipu_s3": DatasetConfig("sipu_s3", "SIPU-s3", true_k=15, loader="sipu",
        loader_kwargs={"stem": "s3"}),
    "sipu_s4": DatasetConfig("sipu_s4", "SIPU-s4", true_k=15, loader="sipu",
        loader_kwargs={"stem": "s4"}),
    "sipu_a1": DatasetConfig("sipu_a1", "SIPU-a1", true_k=20, loader="sipu",
        loader_kwargs={"stem": "a1"}),
    "sipu_R15": DatasetConfig("sipu_R15", "SIPU-R15", true_k=15, loader="sipu",
        loader_kwargs={"stem": "R15"}),
    "sipu_D31": DatasetConfig("sipu_D31", "SIPU-D31", true_k=31, loader="sipu",
        loader_kwargs={"stem": "D31"}),
    "sipu_aggregation": DatasetConfig("sipu_aggregation", "SIPU-Aggregation", true_k=7,
        loader="sipu", loader_kwargs={"stem": "Aggregation"}),
    "sipu_compound": DatasetConfig("sipu_compound", "SIPU-Compound", true_k=6,
        loader="sipu", loader_kwargs={"stem": "Compound"}),
    "sipu_flame": DatasetConfig("sipu_flame", "SIPU-Flame", true_k=2, loader="sipu",
        loader_kwargs={"stem": "flame"}),
    "sipu_jain": DatasetConfig("sipu_jain", "SIPU-Jain", true_k=2, loader="sipu",
        loader_kwargs={"stem": "jain"}),
    "sipu_spiral": DatasetConfig("sipu_spiral", "SIPU-Spiral", true_k=3, loader="sipu",
        loader_kwargs={"stem": "spiral"}),
    "sipu_path_based": DatasetConfig("sipu_path_based", "SIPU-path-based", true_k=3,
        loader="sipu", loader_kwargs={"stem": "pathbased"}),
    "sipu_unbalance": DatasetConfig("sipu_unbalance", "SIPU-unbalance", true_k=8,
        loader="sipu", loader_kwargs={"stem": "unbalance"}),
}


def _apply_scaler(X, mode):
    if mode == "standard":
        return StandardScaler().fit_transform(X).astype(np.float32)
    if mode == "center":
        return StandardScaler(with_std=False).fit_transform(X).astype(np.float32)
    return X.astype(np.float32)


def _load_sklearn_builtin(cfg):
    import sklearn.datasets as skd
    bunch = getattr(skd, cfg.loader_kwargs["fn"])()
    X = _apply_scaler(bunch.data.astype(np.float32), cfg.scale)
    return X, bunch.target.astype(int)


def _load_openml(cfg):
    kw = dict(cfg.loader_kwargs)
    try:
        X_raw, y_raw = fetch_openml(return_X_y=True, as_frame=False, **kw)
        X = np.asarray(X_raw, dtype=np.float32)
    except (ValueError, TypeError):
        X_df, y_raw = fetch_openml(return_X_y=True, as_frame=True, **kw)
        for col in X_df.columns:
            if X_df[col].dtype == "bool" or X_df[col].dtype == "boolean":
                X_df[col] = X_df[col].astype(int)
            elif X_df[col].dtype.kind in ("O", "U", "S") or hasattr(X_df[col], "cat"):
                X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
        X = X_df.to_numpy(dtype=np.float32, na_value=np.nan)
        if hasattr(y_raw, "to_numpy"):
            y_raw = y_raw.to_numpy()

    for c in np.where(np.any(np.isnan(X), axis=0))[0]:
        X[np.isnan(X[:, c]), c] = float(np.nanmedian(X[:, c]))

    if hasattr(y_raw, 'dtype') and y_raw.dtype.kind in ("U", "S", "O"):
        y = LabelEncoder().fit_transform(y_raw)
    else:
        try:
            y = np.asarray(y_raw).astype(int)
        except (ValueError, TypeError):
            y = LabelEncoder().fit_transform(np.asarray(y_raw).astype(str))
    return _apply_scaler(X, cfg.scale), y


def _download(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    downloading {url}")
    urllib.request.urlretrieve(url, dest)


def _load_sipu(cfg):
    stem = cfg.loader_kwargs["stem"]
    cache = SIPU_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    data_path = cache / f"{stem}.txt"
    _download(f"{SIPU_BASE_URL}/{stem}.txt", data_path)
    raw = np.loadtxt(data_path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    y = None
    if raw.shape[1] >= 2:
        last = raw[:, -1]
        if np.allclose(last, np.round(last)) and len(np.unique(last)) <= max(50, len(raw) // 50):
            y = (last.astype(int) - 1)
            raw = raw[:, :-1]
            print(f"    labels in the file, {len(np.unique(y))} classes")

    X = raw.astype(np.float32)

    cb_stem = stem.lower()
    cb_url = f"https://github.com/gagolews/clustering-data-v1/raw/v1.1.0/sipu/{cb_stem}.labels0.gz"
    cb_path = cache / f"{cb_stem}.labels0.gz"
    if not cb_path.exists():
        try:
            _download(cb_url, cb_path)
        except Exception:
            if cb_path.exists():
                cb_path.unlink()
    if cb_path.exists():
        try:
            import gzip
            with gzip.open(cb_path, 'rt') as f:
                labels = np.array([int(line.strip()) for line in f if line.strip()], dtype=int)
            y = labels - labels.min()
            print(f"    labels from clustering-benchmarks, {len(np.unique(y))} classes")
        except Exception:
            pass

    if y is None:
        print(f"    no labels for {stem}")

    return _apply_scaler(X, cfg.scale), y


def load_dataset(cfg, random_state=42):
    print(f"\n{cfg.name}")

    if cfg.loader == "sklearn_builtin":
        X, y = _load_sklearn_builtin(cfg)
    elif cfg.loader == "openml":
        X, y = _load_openml(cfg)
    elif cfg.loader == "sipu":
        X, y = _load_sipu(cfg)
    else:
        raise ValueError(f"Unknown loader: {cfg.loader!r}")

    if cfg.pca_dims is not None and X.shape[1] > cfg.pca_dims:
        print(f"    PCA: {X.shape[1]}D -> {cfg.pca_dims}D")
        X = PCA(n_components=cfg.pca_dims, random_state=random_state).fit_transform(X).astype(np.float32)

    n_cls = len(np.unique(y)) if y is not None else "?"
    print(f"    {X.shape[0]} samples, {X.shape[1]} features, {n_cls} classes")
    return X, y
