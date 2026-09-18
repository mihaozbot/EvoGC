"""Evolving microcluster graph consensus clustering.

Ozbot and Skrjanc, Evolving Beyond Gaussian Prototypes with Multi-Scale Graph
Consensus on Microclusters, IEEE EAIS 2026.

    model = EvoGC(K=10, random_state=0).fit(X)
    labels = model.pooled_row["labels"]
"""
from __future__ import annotations
import os, warnings, math, time, heapq, threading
from collections import defaultdict
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, csc_matrix, diags as sp_diags
from scipy.sparse.linalg import eigsh as _eigsh, LinearOperator, lobpcg as _lobpcg
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage as _scipy_linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

C_TARGET_DEFAULT = 5000
MLE_SUBSAMPLE = 2000
ISO_MERGE_K = 10
ISO_THRESHOLD_RATIO = 0.1
EPS = 1e-10
WORKER_TIMEOUT = 3600
MIN_WORKERS = 4
CPU_WORKER_RATIO = 3
C_SINGLE_THREAD = 300
COMP_N_ROUTE = 30
COMP_K_ROUTE = 5
COMP_K_NN = 10
COMP_BATCH_SIZE = 8192
COMP_DIST_RING = 2000
KSWEEP_K_MIN = 3
LOBPCG_MAXITER = 200
LOBPCG_TOL = 1e-6
LINKAGE_MAX_PTS = 200
ISO_FRAC = 0.01
PEAK_DIVERSITY = 0.4
SEED_MAX = 2 ** 31 - 1


def sqdist(A, B):
    A, B = np.asarray(A, dtype=np.float32), np.asarray(B, dtype=np.float32)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    D = np.sum(A * A, 1, keepdims=True) + np.sum(B * B, 1, keepdims=True).T - 2 * (A @ B.T)
    return np.maximum(D, 0).astype(np.float32)


def estimate_intrinsic_dim_mle(X, k=None):
    X, n = np.asarray(X, dtype=np.float32), len(X)
    if n < 4:
        return 1.0
    k = max(2, min(int(round(n ** (1 / 3))), 20, n - 1)) if k is None else max(2, min(k, n - 1))
    dists = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, 1:]
    dists = np.maximum(dists, dists[dists > 0].min() if np.any(dists > 0) else 1e-10)
    s = np.log(dists[:, -1:] / dists[:, :-1]).sum(axis=1)
    valid = s > 1e-10
    return max(float(np.median((k - 1) / s[valid])), 1.0) if valid.sum() >= k else 1.0


def build_theoretical_k_sweep(C, n_samples=None, d_int=None):
    if C <= KSWEEP_K_MIN:
        return []
    k_min = max(KSWEEP_K_MIN, int(d_int)) if d_int and d_int > KSWEEP_K_MIN else KSWEEP_K_MIN
    k_sqrt = int(math.sqrt(C))
    k_max = max(k_min, min(int(math.sqrt(n_samples)), k_sqrt, C - 1)) if n_samples and n_samples > C else max(k_min, min(k_sqrt, C - 1))
    full = list(range(k_min, k_max + 1))
    pts = max(8, min(10, int(math.sqrt(k_max))))
    if len(full) <= pts:
        return full
    return sorted(set(int(g) for g in np.geomspace(k_min, k_max, pts).astype(int) if k_min <= g <= k_max) | {k_max})


def clustering_accuracy(y_true, y_pred):
    yt = np.unique(np.asarray(y_true), return_inverse=True)[1]
    yp = np.unique(np.asarray(y_pred), return_inverse=True)[1]
    C = max(yt.max() + 1, yp.max() + 1)
    M = np.zeros((C, C), dtype=np.int64)
    np.add.at(M, (yp, yt), 1)
    ri, ci = linear_sum_assignment(M.max() - M)
    return float(M[ri, ci].sum() / len(y_true))


def precompute_knn(X, k_max):
    k_use = min(int(k_max) + 1, len(X))
    d, i = NearestNeighbors(n_neighbors=k_use).fit(X).kneighbors(X)
    return d[:, 1:], i[:, 1:]


def build_snn_graphs(X, knn_indices, k, counts=None):
    n = X.shape[0]
    k = min(k, knn_indices.shape[1])
    knn_idx = knn_indices[:, :k]
    row_ind = np.repeat(np.arange(n), k)
    col_ind = knn_idx.ravel()
    B = csr_matrix((np.ones(n * k, dtype=np.float64), (row_ind, col_ind)), shape=(n, n))
    SNN = (B @ B.T).tocsr()
    edge_mask = (B + B.T.tocsr())
    edge_mask.data[:] = 1.0
    W_sim = (SNN.multiply(edge_mask) / k).tocsr()
    W_sim = ((W_sim + W_sim.T) * 0.5).tocsr()
    W_sim.setdiag(0.0)
    W_sim.eliminate_zeros()
    if counts is not None and np.any(counts != 1):
        c_f = np.maximum(counts.astype(np.float64), 1.0)
        inv_mean = 1.0 / max(c_f.mean(), 1e-10)
        D_boost = sp_diags(c_f * inv_mean)
        W_sim = (D_boost @ W_sim @ D_boost).tocsr()
        W_sim.setdiag(0.0)
        W_sim.eliminate_zeros()
    return W_sim, knn_idx


try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _linkage_kernel_numba(flat_pts, offsets, knn_idx, k_use, centers):
        C, d = len(offsets) - 1, flat_pts.shape[1]
        result = np.full((C, k_use), np.inf)
        for idx in prange(C * k_use):
            i, j_pos = idx // k_use, idx % k_use
            j = knn_idx[i, j_pos]
            si, ei, sj, ej = offsets[i], offsets[i + 1], offsets[j], offsets[j + 1]
            if ei - si == 0 or ej - sj == 0:
                sq = 0.0
                for dd in range(d):
                    diff = centers[i, dd] - centers[j, dd]
                    sq += diff * diff
                result[i, j_pos] = np.sqrt(sq)
                continue
            best = np.inf
            for a in range(ei - si):
                for b in range(ej - sj):
                    sq = 0.0
                    for dd in range(d):
                        diff = flat_pts[si + a, dd] - flat_pts[sj + b, dd]
                        sq += diff * diff
                    if sq < best:
                        best = sq
            result[i, j_pos] = np.sqrt(best)
        return result

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _linkage_kernel_python(flat_pts, offsets, knn_idx, k_use, centers):
    C = len(offsets) - 1
    result = np.full((C, k_use), np.inf, dtype=np.float64)
    for i in range(C):
        si, ei = int(offsets[i]), int(offsets[i + 1])
        pi = flat_pts[si:ei]
        for jp in range(k_use):
            j = knn_idx[i, jp]
            sj, ej = int(offsets[j]), int(offsets[j + 1])
            pj = flat_pts[sj:ej]
            if len(pi) == 0 or len(pj) == 0:
                result[i, jp] = float(np.sqrt(((centers[i] - centers[j]) ** 2).sum()))
            else:
                result[i, jp] = float(cdist(pi, pj).min())
    return result


def compute_linkage_distances(X_raw, mc_labels, centers, knn_indices, k_max, max_pts_per_mc=LINKAGE_MAX_PTS):
    C, d = centers.shape
    k_use = min(k_max, knn_indices.shape[1])
    mc_labels_arr = np.asarray(mc_labels)
    order = np.argsort(mc_labels_arr)
    sorted_labels = mc_labels_arr[order]
    X_sorted = X_raw[order].astype(np.float32)
    breaks = np.searchsorted(sorted_labels, np.arange(C))
    ends = np.append(breaks[1:], len(order))
    sizes = np.minimum(ends - breaks, max_pts_per_mc).astype(np.int64)
    offsets = np.zeros(C + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    flat_pts = np.empty((int(offsets[-1]), d), dtype=np.float32)
    for j in range(C):
        s_src, n_pts = int(breaks[j]), int(sizes[j])
        if n_pts > 0:
            e_src = int(ends[j])
            if e_src - s_src > max_pts_per_mc:
                idx = np.random.RandomState(j).choice(e_src - s_src, max_pts_per_mc, replace=False)
                flat_pts[offsets[j]:offsets[j+1]] = X_sorted[s_src + idx]
            else:
                flat_pts[offsets[j]:offsets[j+1]] = X_sorted[s_src:e_src]
    knn_idx = knn_indices[:, :k_use].astype(np.int32)
    centers_f32 = centers.astype(np.float32)
    if _HAS_NUMBA:
        return _linkage_kernel_numba(flat_pts, offsets, knn_idx, k_use, centers_f32)
    return _linkage_kernel_python(flat_pts, offsets, knn_idx, k_use, centers_f32)


def evolving_compress(X, C_target=C_TARGET_DEFAULT, random_state=42, radius_scale=1.0):

    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    INIT_SIZE = min(max(200, int(n * 0.1)), n)
    lnN = math.log(max(n, 2))
    _rng = np.random.RandomState(random_state if random_state is not None else 42)
    _perm = _rng.permutation(n)
    X = X[_perm]

    n_r = min(COMP_N_ROUTE, INIT_SIZE, n)
    k_r = min(COMP_K_ROUTE, n_r)
    init_pool = X[_rng.choice(min(INIT_SIZE, n), size=min(INIT_SIZE, n), replace=False)]
    rc = np.zeros((n_r, d), dtype=np.float32)
    rc[0] = init_pool[_rng.randint(0, len(init_pool))]
    for i in range(1, n_r):
        D = ((init_pool[:, None, :] - rc[None, :i, :]) ** 2).sum(2).min(1)
        rc[i] = init_pool[int(np.argmax(D))]
    rc_sum = rc.astype(np.float64).copy()
    rc_cnt = np.ones(n_r, dtype=np.int64)

    max_mpc = n // n_r + n // 5
    mc_cen = np.zeros((n_r * max_mpc, d), dtype=np.float32)
    mc_cnt = np.zeros(n_r * max_mpc, dtype=np.int32)
    mc_nrm = np.zeros(n_r * max_mpc, dtype=np.float32)
    cell_mc = np.zeros(n_r, dtype=np.int32)
    cell_off = np.arange(n_r, dtype=np.int32) * max_mpc
    labels = np.full(n, -1, dtype=np.int32)

    radius_sq = -1.0
    mc_size = 1
    alpha = 0.0
    d_int = float(d)
    median_nn1 = 0.0
    _params_set = False
    mc_radius_sq = None
    dist_ring = []
    _last_update = 0
    _samples_seen = 0

    _batch_starts = [(0, INIT_SIZE)] if INIT_SIZE < n else [(0, n)]
    if INIT_SIZE < n:
        _batch_starts += [(s, min(s + COMP_BATCH_SIZE, n)) for s in range(INIT_SIZE, n, COMP_BATCH_SIZE)]

    for start, end in _batch_starts:
        batch = X[start:end]
        bsz = len(batch)
        bn = (batch * batch).sum(1)

        rcn = (rc * rc).sum(1)
        D_route = bn[:, None] + rcn[None, :] - 2 * batch @ rc.T
        top_k = np.argpartition(D_route, k_r, axis=1)[:, :k_r]
        ca = np.argmin(D_route, axis=1)

        cell_counts_batch = np.bincount(ca, minlength=n_r)
        np.add.at(rc_cnt, np.arange(n_r), cell_counts_batch)
        batch_f64 = batch.astype(np.float64)
        for ci in np.where(cell_counts_batch > 0)[0]:
            rc_sum[ci] += batch_f64[ca == ci].sum(0)
            rc[ci] = (rc_sum[ci] / rc_cnt[ci]).astype(np.float32)

        best_mc = np.full(bsz, -1, dtype=np.int32)
        best_dist = np.full(bsz, np.inf, dtype=np.float32)
        for ci in range(n_r):
            mc = int(cell_mc[ci])
            if mc == 0:
                continue
            m = np.any(top_k == ci, axis=1)
            if not m.any():
                continue
            idx = np.where(m)[0]
            off = cell_off[ci]
            Dm = bn[idx, None] + mc_nrm[off:off + mc][None, :] - 2 * batch[idx] @ mc_cen[off:off + mc].T
            jl = np.argmin(Dm, axis=1)
            dl = Dm[np.arange(len(idx)), jl]
            better = dl < best_dist[idx]
            bi = idx[better]
            best_dist[bi] = dl[better]
            best_mc[bi] = off + jl[better]

        mc_cnt_check = mc_cnt[np.maximum(best_mc, 0)]
        if _params_set and mc_radius_sq is not None:
            per_mc_rsq = mc_radius_sq[np.maximum(best_mc, 0)]
            per_mc_rsq[per_mc_rsq <= 0] = radius_sq
            can_abs = (best_dist <= per_mc_rsq) & (best_mc >= 0) & (mc_cnt_check < mc_size)
        else:
            can_abs = (best_dist <= radius_sq) & (best_mc >= 0) & (mc_cnt_check < mc_size)

        ai = np.where(can_abs)[0]
        if len(ai) > 0:
            am = best_mc[ai]
            labels[start + ai] = am
            abs_batch = batch[ai].astype(np.float64)
            uniq_mc, inv_mc = np.unique(am, return_inverse=True)
            old_counts = mc_cnt[uniq_mc].copy()
            np.add.at(mc_cnt, am, 1)
            sums = np.zeros((len(uniq_mc), d), dtype=np.float64)
            np.add.at(sums, inv_mc, abs_batch)
            new_counts = mc_cnt[uniq_mc]
            for ii, mi in enumerate(uniq_mc):
                mc_cen[mi] = ((mc_cen[mi].astype(np.float64) * old_counts[ii] + sums[ii]) / new_counts[ii]).astype(np.float32)
                mc_nrm[mi] = (mc_cen[mi] * mc_cen[mi]).sum()

        bi = np.where(~can_abs)[0]
        if len(bi) > 0:
            bi_cells = ca[bi]
            for ci in np.unique(bi_cells):
                w = bi[bi_cells == ci]
                nb = len(w)
                off = cell_off[ci]
                mc = int(cell_mc[ci])
                avail = max_mpc - mc
                if nb > avail:
                    overflow = w[avail:]
                    w = w[:avail]
                    nb = avail
                    if mc > 0:
                        D_ov = bn[overflow, None] + mc_nrm[off:off + mc][None, :] - 2 * batch[overflow] @ mc_cen[off:off + mc].T
                        nearest = off + np.argmin(D_ov, axis=1)
                        ov_batch = batch[overflow].astype(np.float64)
                        for oi_idx, nj in enumerate(nearest):
                            nj = int(nj)
                            old = mc_cnt[nj]
                            mc_cnt[nj] = old + 1
                            mc_cen[nj] = ((mc_cen[nj].astype(np.float64) * old + ov_batch[oi_idx]) / (old + 1)).astype(np.float32)
                            mc_nrm[nj] = (mc_cen[nj] * mc_cen[nj]).sum()
                            labels[start + overflow[oi_idx]] = nj
                if nb > 0:
                    mc_cen[off + mc:off + mc + nb] = batch[w]
                    mc_cnt[off + mc:off + mc + nb] = 1
                    mc_nrm[off + mc:off + mc + nb] = bn[w]
                    labels[start + w] = np.arange(off + mc, off + mc + nb, dtype=np.int32)
                    cell_mc[ci] = mc + nb

        _samples_seen += bsz

        if not _params_set:
            finite = (best_dist < np.inf) & (best_mc >= 0)
            if finite.any():
                dist_ring.extend(np.sqrt(np.maximum(best_dist[finite], 0.0)).tolist())
        else:
            if can_abs.any():
                dist_ring.extend(np.sqrt(np.maximum(best_dist[can_abs], 0.0)).tolist())
        if len(dist_ring) > COMP_DIST_RING:
            dist_ring = dist_ring[-COMP_DIST_RING:]

        if not _params_set and _samples_seen >= INIT_SIZE:

            live_mask = np.zeros(n_r * max_mpc, dtype=bool)
            for ci in range(n_r):
                o = cell_off[ci]
                live_mask[o:o + cell_mc[ci]] = True
            genesis_centers = mc_cen[live_mask]
            flat_to_global = np.where(live_mask)[0]
            n_gen = len(genesis_centers)
            k_local = min(COMP_K_NN, n_gen - 1)
            dists_gen = NearestNeighbors(n_neighbors=k_local + 1).fit(genesis_centers).kneighbors(genesis_centers)[0][:, 1:]
            min_pos = float(dists_gen[dists_gen > 0].min()) if np.any(dists_gen > 0) else 1e-10
            dists_safe = np.maximum(dists_gen, min_pos)
            sum_log = np.log(dists_safe[:, -1:] / dists_safe[:, :-1]).sum(axis=1)
            valid = sum_log > 1e-10
            mc_d_int = np.full(n_gen, float(d))
            mc_d_int[valid] = np.maximum(float(k_local - 1) / sum_log[valid], 1.0)
            mc_alpha = np.maximum((mc_d_int - 1.0) / 2.0, 0.0)
            mc_alpha[mc_d_int < 2.0] *= np.maximum(0.0, mc_d_int[mc_d_int < 2.0] - 1.0)
            global_alpha = max((float(np.median(mc_d_int)) - 1.0) / 2.0, 0.1)
            mc_alpha = np.maximum(mc_alpha, global_alpha * 0.5)
            mc_radius_sq_flat = (dists_gen[:, 0] * mc_alpha * radius_scale) ** 2

            mc_radius_sq = np.zeros(n_r * max_mpc, dtype=np.float32)
            mc_radius_sq[flat_to_global] = mc_radius_sq_flat

            d_int = float(np.median(mc_d_int))
            alpha = max((d_int - 1.0) / 2.0, 0.0)
            if d_int < 2.0:
                alpha *= max(0.0, d_int - 1.0)
            median_nn1 = float(np.median(dist_ring)) if len(dist_ring) > 20 else (float(np.median(dists_gen[:, 0])) if k_local >= 2 else 1.0)
            radius_sq = (median_nn1 * alpha * radius_scale) ** 2
            mc_size = max(1, min(int(alpha * lnN), math.ceil(lnN)))
            _params_set = True

        elif _params_set and (_samples_seen - _last_update) >= COMP_DIST_RING // 2 and len(dist_ring) > 20:
            _last_update = _samples_seen
            median_nn1 = float(np.median(dist_ring))
            radius_sq = (median_nn1 * alpha * radius_scale) ** 2

    live = mc_cnt[:n_r * max_mpc] > 0

    for ci in range(n_r):
        o = cell_off[ci]
        live[o + cell_mc[ci]:o + max_mpc] = False
    C = int(live.sum())
    remap = np.full(n_r * max_mpc, -1, dtype=np.int32)
    remap[live] = np.arange(C, dtype=np.int32)
    labels = remap[labels]
    centers = mc_cen[live].copy()
    counts = mc_cnt[live].copy()

    order = np.argsort(labels)
    sorted_labels = labels[order]
    breaks = np.searchsorted(sorted_labels, np.arange(C))
    ends = np.append(breaks[1:], len(order))
    for j in range(C):
        s, e = int(breaks[j]), int(ends[j])
        if e <= s:
            continue
        members = order[s:e]
        pts = X[members]
        mean = pts.mean(axis=0)
        centers[j] = pts[np.argmin(((pts - mean) ** 2).sum(axis=1))]

    first_appear = np.full(C, n, dtype=np.int32)
    np.minimum.at(first_appear, labels, np.arange(n, dtype=np.int32))
    order = np.argsort(first_appear)
    inv_order = np.empty(C, dtype=np.int32)
    inv_order[order] = np.arange(C, dtype=np.int32)
    labels = inv_order[labels]
    centers = centers[order]
    counts = counts[order]

    labels_orig = np.empty(n, dtype=np.int32)
    labels_orig[_perm] = labels
    return labels_orig, centers, counts, {'d_int': d_int, 'alpha': alpha, 'mc_size': mc_size}


def _impl_peaks_labels(order, knn_idx, rho, Xc):
    C, D, k = order.shape[0], Xc.shape[1], knn_idx.shape[1]
    parent = np.full(C, -1, dtype=np.int32)
    for oi in range(C):
        idx = order[oi]
        for j in range(k):
            neigh = knn_idx[idx, j]
            if neigh >= 0 and neigh < C and rho[neigh] > rho[idx]:
                parent[idx] = np.int32(neigh)
                break
    peak_ids = np.array([i for i in range(C) if parent[i] < 0], dtype=np.int32)
    n_peaks = len(peak_ids)
    sub_label = np.full(C, -1, dtype=np.int32)
    for si in range(n_peaks):
        sub_label[peak_ids[si]] = np.int32(si)
    for oi in range(C):
        idx = order[oi]
        if sub_label[idx] >= 0:
            continue
        cur = idx
        for _ in range(C):
            if parent[cur] < 0 or sub_label[cur] >= 0:
                break
            cur = parent[cur]
        if sub_label[cur] >= 0:
            lbl = sub_label[cur]
            c = idx
            for _ in range(C):
                if sub_label[c] >= 0:
                    break
                sub_label[c] = lbl
                if parent[c] >= 0:
                    c = parent[c]
                else:
                    break
        else:
            min_d2, best = np.inf, np.int32(0)
            for pj in range(n_peaks):
                d2 = 0.0
                for dim in range(D):
                    diff = float(Xc[idx, dim]) - float(Xc[peak_ids[pj], dim])
                    d2 += diff * diff
                if d2 < min_d2:
                    min_d2 = d2
                    best = np.int32(pj)
            sub_label[idx] = best
    return peak_ids, sub_label


def _impl_connectivity(knn_idx, sub_label, rho, peak_ids, k_b):
    C, n_peaks = sub_label.shape[0], peak_ids.shape[0]
    kb = min(k_b, knn_idx.shape[1])
    connect = np.zeros((n_peaks, n_peaks), dtype=np.bool_)
    ascent_rho = np.zeros(n_peaks, dtype=np.float64)
    for i in range(C):
        si = sub_label[i]
        if si < 0:
            continue
        for jc in range(kb):
            nb = knn_idx[i, jc]
            if nb < 0 or nb >= C:
                continue
            sj = sub_label[nb]
            if si == sj or sj < 0:
                continue
            is_mutual = False
            for m in range(kb):
                if knn_idx[nb, m] == i:
                    is_mutual = True
                    break
            if not is_mutual:
                continue
            connect[si, sj] = True
            connect[sj, si] = True
            if rho[peak_ids[sj]] > rho[peak_ids[si]] and rho[nb] > ascent_rho[si]:
                ascent_rho[si] = rho[nb]
            if rho[peak_ids[si]] > rho[peak_ids[sj]] and rho[i] > ascent_rho[sj]:
                ascent_rho[sj] = rho[i]
    return connect, ascent_rho


def _impl_delta_atten(peak_ids, rho, ascent_border_rho, Xc, region_labels, region_peak_idx):
    n_peaks, n_regions, D = peak_ids.shape[0], region_peak_idx.shape[0], Xc.shape[1]
    delta = np.zeros(n_peaks, dtype=np.float64)
    for pi in range(n_peaks):
        p = peak_ids[pi]
        r = region_labels[pi]
        if pi == region_peak_idx[r]:
            delta[pi] = -1.0
            continue
        min_dist = np.inf
        for pj in range(n_peaks):
            if region_labels[pj] != r:
                continue
            if rho[peak_ids[pj]] <= rho[p]:
                continue
            d2 = 0.0
            for dim in range(D):
                diff = float(Xc[p, dim]) - float(Xc[peak_ids[pj], dim])
                d2 += diff * diff
            d = np.sqrt(d2)
            if d < min_dist:
                min_dist = d
        if min_dist < np.inf:
            delta[pi] = min_dist
    max_delta = 0.0
    for pi in range(n_peaks):
        if delta[pi] > max_delta:
            max_delta = delta[pi]
    for r in range(n_regions):
        rp = region_peak_idx[r]
        delta[rp] = max_delta * 2 if max_delta > 0 else 1.0
    if n_regions == 1:
        rp = region_peak_idx[0]
        max_non_rp = 0.0
        for pi in range(n_peaks):
            if pi != rp and delta[pi] > max_non_rp:
                max_non_rp = delta[pi]
        delta[rp] = max_non_rp * 2 if max_non_rp > 0 else 1.0
    for pi in range(n_peaks):
        rho_p = max(rho[peak_ids[pi]], 1e-10)
        atten = max(0.0, 1.0 - ascent_border_rho[pi] / rho_p)
        delta[pi] = delta[pi] * atten
    return delta


def _impl_internal_stats(indptr, indices, data, mask):
    tw, ne = 0.0, 0
    for i in range(mask.shape[0]):
        if not mask[i]:
            continue
        for jp in range(indptr[i], indptr[i + 1]):
            if indices[jp] != i and mask[indices[jp]]:
                tw += data[jp]
                ne += 1
    return tw * 0.5, tw / (ne + 1e-10) if ne > 0 else 0.0


def _impl_chameleon_score(indptr, indices, data, mask_a, mask_b, int_cut_a, int_cut_b, icl_a, icl_b, alpha, beta):
    cut, ne = 0.0, 0
    for i in range(mask_a.shape[0]):
        if not mask_a[i]:
            continue
        for jp in range(indptr[i], indptr[i + 1]):
            if mask_b[indices[jp]]:
                cut += data[jp]
                ne += 1
    if cut < 1e-10:
        return 0.0
    ri = cut / (0.5 * (int_cut_a + int_cut_b) + 1e-10)
    rc = (cut / (ne + 1e-10)) / (0.5 * (icl_a + icl_b) + 1e-10)
    return (ri ** alpha) * (rc ** beta)


try:
    from numba import njit as _njit
    _nb_peaks_labels = _njit(cache=True)(_impl_peaks_labels)
    _nb_connectivity = _njit(cache=True)(_impl_connectivity)
    _nb_delta_atten = _njit(cache=True)(_impl_delta_atten)
    _nb_internal_stats = _njit(cache=True)(_impl_internal_stats)
    _nb_chameleon_score = _njit(cache=True)(_impl_chameleon_score)
except ImportError:
    _nb_peaks_labels = _impl_peaks_labels
    _nb_connectivity = _impl_connectivity
    _nb_delta_atten = _impl_delta_atten
    _nb_internal_stats = _impl_internal_stats
    _nb_chameleon_score = _impl_chameleon_score


def chameleon_greedy_merge(W_sim, sub_labels, K_target, Xc, counts, merge_noise=0.0, seed=42):
    eps = 1e-10
    C = len(sub_labels)
    n_sub = int(sub_labels.max()) + 1
    if n_sub <= K_target:
        return sub_labels.copy()
    _merge_rng = np.random.RandomState(seed) if merge_noise > 0 else None
    W_csr = W_sim.tocsr().astype(np.float64)
    _indptr = W_csr.indptr.astype(np.int32)
    _indices = W_csr.indices.astype(np.int32)
    _data = W_csr.data.astype(np.float64)
    groups, masks = {}, {}
    order_sl = np.argsort(sub_labels)
    sorted_sl = sub_labels[order_sl]
    brk = np.searchsorted(sorted_sl, np.arange(n_sub))
    end = np.append(brk[1:], len(order_sl))
    for g in range(n_sub):
        if end[g] > brk[g]:
            m = order_sl[brk[g]:end[g]]
            groups[g] = m
            mask = np.zeros(C, dtype=np.bool_)
            mask[m] = True
            masks[g] = mask
    active = set(groups.keys())
    rows_sp, cols_sp = W_csr.nonzero()
    sl_r, sl_c = sub_labels[rows_sp], sub_labels[cols_sp]
    cross_mask = sl_r != sl_c
    adj = defaultdict(set)
    for idx in np.where(cross_mask)[0]:
        adj[int(sl_r[idx])].add(int(sl_c[idx]))
        adj[int(sl_c[idx])].add(int(sl_r[idx]))
    int_cuts, icl_vals = {}, {}

    def _stats(gid):
        mask = masks[gid]
        n = int(mask.sum())
        if n < 2:
            int_cuts[gid] = 0.0
            icl_vals[gid] = 0.0
        else:
            ic, icl = _nb_internal_stats(_indptr, _indices, _data, mask)
            int_cuts[gid] = ic
            icl_vals[gid] = icl

    def _score(a, b):
        return _nb_chameleon_score(_indptr, _indices, _data, masks[a], masks[b],
                                  int_cuts[a], int_cuts[b], icl_vals[a], icl_vals[b], 1.0, 1.0)

    for g in active:
        _stats(g)
    next_gid = n_sub
    heap = []
    scored = set()
    for a in sorted(active):
        for b in adj.get(a, set()):
            if b <= a or (a, b) in scored:
                continue
            scored.add((a, b))
            s = _score(a, b)
            if _merge_rng is not None:
                s *= np.exp(_merge_rng.randn() * merge_noise)
            if s > eps:
                heapq.heappush(heap, (-s, a, b))
    while len(active) > K_target and heap:
        neg_s, a, b = heapq.heappop(heap)
        if a not in active or b not in active:
            continue
        groups[next_gid] = np.concatenate([groups[a], groups[b]])
        masks[next_gid] = masks[a] | masks[b]
        del masks[a], masks[b]
        active.discard(a)
        active.discard(b)
        active.add(next_gid)
        _stats(next_gid)
        nbrs_new = set()
        for old_gid in (a, b):
            for nbr in adj.get(old_gid, set()):
                if nbr == a or nbr == b or nbr not in active:
                    continue
                nbrs_new.add(nbr)
                adj[nbr].discard(a)
                adj[nbr].discard(b)
                adj[nbr].add(next_gid)
            adj.pop(old_gid, None)
        adj[next_gid] = nbrs_new
        for other in nbrs_new:
            s = _score(next_gid, other)
            if _merge_rng is not None:
                s *= np.exp(_merge_rng.randn() * merge_noise)
            if s > eps:
                heapq.heappush(heap, (-s, next_gid, other))
        next_gid += 1
    final = np.full(C, -1, dtype=np.int32)
    for new_gid, gid in enumerate(sorted(active)):
        final[groups[gid]] = new_gid
    bad = final < 0
    if bad.any():
        good = ~bad
        if good.any():
            final[bad] = final[good][np.argmin(sqdist(Xc[bad], Xc[good]), axis=1)]
        else:
            final[bad] = 0
    return final


def _rmdpc_select_peaks(Xc, knn_idx, knn_dists, K_target, seed=42, knn_idx_full=None, knn_dists_full=None):
    C = Xc.shape[0]
    if K_target >= C:
        return np.arange(C, dtype=np.int32), {
            "peak_ids": np.arange(C), "gamma": np.ones(C),
            "sub_label": np.arange(C, dtype=np.int32), "n_peaks": C}
    k_rmdpc = min(max(2, int(np.ceil(np.sqrt(C)))), C - 1)
    _kd_rho = knn_dists_full if knn_dists_full is not None else knn_dists
    k_rho = min(k_rmdpc, _kd_rho.shape[1])
    rho = 1.0 / np.maximum(_kd_rho[:, :k_rho].mean(axis=1), 1e-10)
    _ki_peaks = knn_idx_full if knn_idx_full is not None else knn_idx
    k_peaks = min(k_rmdpc, _ki_peaks.shape[1])
    knn_idx_r = _ki_peaks[:, :k_peaks].astype(np.int32)
    k = k_peaks
    k_b = max(1, k // 2)
    order = np.argsort(-rho)
    Xc_f32 = np.ascontiguousarray(Xc, dtype=np.float32)
    peak_ids, sub_label = _nb_peaks_labels(order, knn_idx_r, rho, Xc_f32)
    n_peaks = len(peak_ids)
    if n_peaks < 2:
        rng = np.random.RandomState(seed)
        rho_fb = np.maximum(rho.astype(np.float64), 1e-10)
        selected = [int(np.argmax(rho))]
        for _ in range(K_target - 1):
            D2 = sqdist(Xc, Xc[selected])
            min_d2 = D2.min(axis=1).astype(np.float64)
            min_d2[selected] = 0.0
            weighted = rho_fb * min_d2
            total = weighted.sum()
            if total < 1e-12:
                unseen = list(set(range(C)) - set(selected))
                selected.append(unseen[0] if unseen else 0)
            else:
                selected.append(int(rng.choice(C, p=weighted / total)))
        fallback = np.array(selected, dtype=np.int32)
        return fallback, {"peak_ids": fallback, "gamma": np.ones(len(fallback)),
                          "sub_label": np.zeros(C, dtype=np.int32), "n_peaks": max(1, n_peaks)}
    connect, ascent_border_rho = _nb_connectivity(knn_idx_r, sub_label, rho, peak_ids, np.int32(k_b))
    region_id = np.arange(n_peaks, dtype=np.int32)

    def _find(x):
        while region_id[x] != x:
            region_id[x] = region_id[region_id[x]]
            x = region_id[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            region_id[ra] = rb

    for i in range(n_peaks):
        for j in range(i + 1, n_peaks):
            if connect[i, j]:
                _union(i, j)
    region_labels = np.array([_find(i) for i in range(n_peaks)], dtype=np.int32)
    _ru, region_labels = np.unique(region_labels, return_inverse=True)
    n_regions = len(_ru)
    point_region = region_labels[sub_label]
    rho_norm = rho.copy()
    for r in range(n_regions):
        mask_r = point_region == r
        if np.any(mask_r):
            rho_norm[mask_r] /= max(rho[mask_r].max(), 1e-10)
    region_peak_idx = np.full(n_regions, -1, dtype=np.int32)
    for r in range(n_regions):
        peaks_r = np.where(region_labels == r)[0]
        region_peak_idx[r] = peaks_r[np.argmax(rho[peak_ids[peaks_r]])]
    delta = _nb_delta_atten(peak_ids, rho, ascent_border_rho, Xc_f32,
                            region_labels.astype(np.int32), region_peak_idx.astype(np.int32))
    gamma = rho_norm[peak_ids] * delta
    n_select = min(2 * K_target, n_peaks)
    keep_peak_idx = set(np.argsort(-gamma)[:n_select].tolist())

    if len(keep_peak_idx) < n_peaks:
        remap = np.full(n_peaks, -1, dtype=np.int32)
        for pi in keep_peak_idx:
            remap[pi] = pi

        removed = [pi for pi in range(n_peaks) if pi not in keep_peak_idx]
        kept_list = sorted(keep_peak_idx)
        D_rm = sqdist(Xc_f32[peak_ids[removed]], Xc_f32[peak_ids[kept_list]])
        for i, pi in enumerate(removed):
            remap[pi] = kept_list[int(np.argmin(D_rm[i]))]
        sub_label = remap[sub_label]
        _, sub_label = np.unique(sub_label, return_inverse=True)
        sub_label = sub_label.astype(np.int32)
        n_peaks = int(sub_label.max()) + 1
    return peak_ids[sorted(keep_peak_idx)].astype(np.int32), {
        "peak_ids": peak_ids, "gamma": gamma, "sub_label": sub_label, "n_peaks": n_peaks}


def _compute_diffusion_features(centers, knn_i, d_int, d_embed, C, random_state):
    k_diff = min(max(3, int(d_int) + 2), C - 1)
    W, _ = build_snn_graphs(centers, knn_i[:, :k_diff], k_diff)
    W_sym = ((W + W.T) * 0.5).tocsr().astype(np.float64)
    W_sym.setdiag(0.0)
    W_sym.eliminate_zeros()
    deg = np.array(W_sym.sum(axis=1)).ravel()
    P = sp_diags(1.0 / np.maximum(deg, EPS)) @ W_sym
    P2 = (P @ P).tocsr()
    n_svd = min(d_embed + 1, C - 1)
    U, S, _ = randomized_svd(P2, n_components=n_svd, random_state=random_state)
    d_use = min(d_embed, n_svd - 1)
    if d_use <= 0:
        return None, 0
    V = (U[:, 1:d_use + 1] * S[None, 1:d_use + 1]).astype(np.float32)
    orig_var, diff_var = np.var(centers, axis=0).sum(), np.var(V, axis=0).sum()
    if diff_var > EPS:
        V *= np.sqrt(orig_var / diff_var)
    return V, d_use


def _merge_isolated_clusters(centers, labels, counts, C, n, X_f32, iso_frac):
    if C <= ISO_MERGE_K or C >= n:
        return centers, labels, counts, C, 0
    k_merge = min(ISO_MERGE_K, C - 1)
    _, knn_idx_m = precompute_knn(centers, k_merge)
    revknn = np.zeros(C, dtype=np.int32)
    np.add.at(revknn, knn_idx_m.ravel(), 1)
    iso_thresh = max(1, round(ISO_THRESHOLD_RATIO * k_merge))
    isolated = revknn <= iso_thresh
    n_iso = int(isolated.sum())
    iso_mass = int(counts[isolated].sum()) if n_iso else 0
    if n_iso == 0 or n_iso >= C // 2 or iso_mass / max(n, 1) < iso_frac:
        return centers, labels, counts, C, 0
    big_idx, iso_idx = np.where(~isolated)[0], np.where(isolated)[0]
    nearest_big = big_idx[np.argmin(sqdist(centers[iso_idx], centers[big_idx]), axis=1)]
    remap = np.arange(C, dtype=np.int32)
    remap[iso_idx] = nearest_big
    labels = remap[labels]
    new_counts = np.bincount(labels, minlength=C).astype(np.int32)
    alive = new_counts > 0
    if not np.all(alive):
        alive_idx = np.where(alive)[0]
        forward_map = np.full(C, -1, dtype=np.int32)
        forward_map[alive_idx] = np.arange(len(alive_idx), dtype=np.int32)
        labels = forward_map[labels]
        d = X_f32.shape[1]
        C_new = int(alive.sum())
        sums = np.zeros((C_new, d), dtype=np.float64)
        np.add.at(sums, labels, X_f32.astype(np.float64))
        new_counts = np.bincount(labels, minlength=C_new).astype(np.int32)
        centers = (sums / np.maximum(new_counts, 1).astype(np.float64)[:, None]).astype(np.float32)
        counts = new_counts
        C = C_new
    return centers, labels, counts, C, n_iso


def _compress_one_seed(X_f32_orig, C_target, seed, k_max_target):
    for _ek in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[_ek] = "1"
    n, d = X_f32_orig.shape
    rng_s = np.random.RandomState(seed)
    perm = rng_s.permutation(n)
    X_f32 = np.ascontiguousarray(X_f32_orig[perm])
    sub = np.random.RandomState(seed).choice(n, min(n, MLE_SUBSAMPLE), replace=False)
    d_int = estimate_intrinsic_dim_mle(X_f32[sub])
    labels, centers, counts, cinfo = evolving_compress(X_f32, C_target, random_state=seed)
    C = len(centers)
    centers, labels, counts, C, _ = _merge_isolated_clusters(centers, labels, counts, C, n, X_f32, ISO_FRAC)
    k_sweep = build_theoretical_k_sweep(C, n_samples=n, d_int=d_int)
    k_max = max(max(k_sweep), k_max_target)
    knn_d, knn_i = precompute_knn(centers, k_max)
    d_embed = min(d, max(2 * int(round(d_int)), 10), C // 2)
    V_diff = None
    if C > d_embed + 1 and 2 * C <= n:
        V_diff, _ = _compute_diffusion_features(centers, knn_i, d_int, d_embed, C, seed)
    if V_diff is not None:
        knn_d_vd, knn_i_vd = precompute_knn(V_diff, k_max)
    else:
        knn_d = compute_linkage_distances(X_f32, labels, centers, knn_i, k_max)
        knn_d_vd, knn_i_vd = None, None
    return (labels, centers, counts, knn_d, knn_i, C, perm, V_diff, knn_d_vd, knn_i_vd, d_int)


def _spectral_coassoc(member_labels, C, K, member_weights=None, random_state=42):
    M = len(member_labels)
    if member_weights is None:
        member_weights = np.full(M, 1.0 / M)
    Z_list = []
    for i, ml in enumerate(member_labels):
        Z = csc_matrix((np.ones(C), (np.arange(C), ml)), shape=(C, int(ml.max()) + 1))
        Z_list.append((member_weights[i], Z))
    if C <= 2000:
        W = np.zeros((C, C), dtype=np.float64)
        for w, Z in Z_list:
            Zd = Z.toarray()
            W += w * (Zd @ Zd.T)
        d = W.sum(1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0)
        L = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
        vals_all, vecs_all = np.linalg.eigh(L)
        vecs = vecs_all[:, -K:][:, ::-1]
    else:
        ones = np.ones(C, dtype=np.float64)
        d = sum(w * Z.dot(Z.T.dot(ones)) for w, Z in Z_list)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0)
        def _matvec(v):
            x = d_inv_sqrt * v
            y = np.zeros(C, dtype=np.float64)
            for w, Z in Z_list:
                y += w * Z.dot(Z.T.dot(x))
            return d_inv_sqrt * y

        def _matmat(V):
            X = d_inv_sqrt[:, None] * V
            Y = np.zeros_like(V)
            for w, Z in Z_list:
                Y += w * Z.dot(Z.T.dot(X))
            return d_inv_sqrt[:, None] * Y

        L_op = LinearOperator((C, C), matvec=_matvec, matmat=_matmat, dtype=np.float64)
        try:
            rng = np.random.RandomState(random_state)
            X0 = rng.randn(C, K).astype(np.float64)
            vals, vecs = _lobpcg(L_op, X0, largest=True, maxiter=LOBPCG_MAXITER, tol=LOBPCG_TOL, verbosityLevel=0)
            vecs = vecs[:, np.argsort(-vals)[:K]]
        except Exception:
            vals, vecs = _eigsh(L_op, k=K, which='LM')
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms > 0, norms, 1)
    Z_hac = _scipy_linkage(pdist(vecs, metric='euclidean'), method='average')
    return (fcluster(Z_hac, t=K, criterion='maxclust') - 1).astype(np.int32)


def scout_one_k(centers, knn_d, knn_i, counts, k, K_target, seed, V_diff,
                knn_d_full, knn_i_full, peak_diversity=0.0):
    t0 = time.time()
    Xc = V_diff if V_diff is not None else centers
    W_sim, knn_idx_k = build_snn_graphs(Xc, knn_i, k, counts=counts)
    C = Xc.shape[0]

    if K_target is None or K_target < 2:
        lbl = np.zeros(C, dtype=np.int32)
        return {"k": k, "K_det_single": 1, "n_members": 1, "best_lc": lbl, "elapsed": time.time() - t0}

    _, pool_info = _rmdpc_select_peaks(
        Xc, knn_idx_k, knn_d, K_target, seed=seed,
        knn_idx_full=knn_i_full, knn_dists_full=knn_d_full)
    sub_labels = pool_info["sub_label"]
    n_sub = pool_info["n_peaks"]
    bad = sub_labels < 0
    if np.any(bad):
        bad_idx, good_idx = np.where(bad)[0], np.where(~bad)[0]
        if len(good_idx) > 0:
            sub_labels[bad_idx] = sub_labels[good_idx[np.argmin(sqdist(Xc[bad_idx], Xc[good_idx]), axis=1)]]
    _uniq, _first, _inv = np.unique(sub_labels, return_index=True, return_inverse=True)
    n_sub = len(_uniq)
    _order = np.argsort(_first)
    _rank = np.empty(n_sub, dtype=np.int32)
    _rank[_order] = np.arange(n_sub, dtype=np.int32)
    sub_labels = _rank[_inv].astype(np.int32)

    if n_sub > K_target:
        cham_labels = chameleon_greedy_merge(
            W_sim, sub_labels, K_target, Xc, counts,
            merge_noise=peak_diversity, seed=seed)
    else:
        cham_labels = sub_labels.copy()

    return {"k": k, "K_det_single": int(cham_labels.max()) + 1, "n_members": 1,
            "best_lc": cham_labels, "elapsed": time.time() - t0}


class EvoGC:
    def __init__(self, K=None, n_comp=10, random_state=0):
        self.K = K
        self.n_comp = n_comp
        self.random_state = random_state
        self.leaf_centers = self.leaf_counts = self.sample_compact_leaf_ids = None
        self.knn_dists = self.knn_indices = None
        self.best_k = self.d_int = None
        self.per_k_rows = []
        self.pooled_row = self.summary_best = None
        self.t_total = 0.0

    def fit(self, X, y=None, name="dataset"):
        n = X.shape[0]
        t0 = time.time()
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        X_f32_orig = np.asarray(X, dtype=np.float32)
        X = X[perm]
        if y is not None:
            y = np.asarray(y)[perm]
        X_f32 = np.ascontiguousarray(X_f32_orig[perm])

        sub = np.random.RandomState(self.random_state).choice(n, min(n, MLE_SUBSAMPLE), replace=False)
        self.d_int = estimate_intrinsic_dim_mle(X_f32[sub])

        labels, centers, counts, _ = evolving_compress(X_f32, min(n, C_TARGET_DEFAULT), random_state=self.random_state)
        C = len(centers)
        centers, labels, counts, C, _ = _merge_isolated_clusters(centers, labels, counts, C, n, X_f32, ISO_FRAC)
        self.leaf_centers, self.leaf_counts = centers, counts
        self.sample_compact_leaf_ids = labels.copy()
        C = len(self.leaf_centers)

        k_sweep = build_theoretical_k_sweep(C, n_samples=n, d_int=self.d_int)
        if not k_sweep:
            raise RuntimeError("Too few pooled leaves for a valid k sweep.")
        k_max = max(k_sweep)
        K_target = self.K

        comp_futures, comp_seeds, comp_pipelined = None, [], False
        k_max_extra = k_sweep[-1] if k_sweep else 50
        if self.n_comp > 1:
            seed_rng = np.random.RandomState(self.random_state)
            comp_seeds = [(cr, int(seed_rng.randint(0, SEED_MAX))) for cr in range(1, self.n_comp)]
            try:
                from joblib.externals.loky import get_reusable_executor
                saved_env = {}
                for ek in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"):
                    saved_env[ek] = os.environ.get(ek)
                    os.environ[ek] = "1"
                executor = get_reusable_executor(
                    max_workers=min(len(comp_seeds), max(2, (os.cpu_count() or 8) // 2)),
                    timeout=WORKER_TIMEOUT, reuse="auto")
                comp_futures = [
                    executor.submit(_compress_one_seed, X_f32_orig, min(n, C_TARGET_DEFAULT), seed_r, k_max_extra)
                    for _, seed_r in comp_seeds]
                for ek, ev in saved_env.items():
                    if ev is None:
                        os.environ.pop(ek, None)
                    else:
                        os.environ[ek] = ev
                comp_pipelined = True
            except Exception as e:
                raise RuntimeError(f"loky required for n_comp={self.n_comp}: {e}") from e

        knn_dists, knn_indices = precompute_knn(self.leaf_centers, k_max)
        d_embed = min(self.leaf_centers.shape[1], max(2 * int(round(self.d_int)), 10), C // 2)
        V_diff = None
        if C > d_embed + 1 and 2 * C <= n:
            V_diff, _ = _compute_diffusion_features(
                self.leaf_centers, knn_indices, self.d_int, d_embed, C, self.random_state)
        if V_diff is not None:
            knn_dists, knn_indices = precompute_knn(V_diff, k_max)
        else:
            knn_dists = compute_linkage_distances(
                X_f32, self.sample_compact_leaf_ids, self.leaf_centers, knn_indices, k_max)
        self.knn_dists, self.knn_indices = knn_dists, knn_indices

        bg_reps, bg_futs = [], []
        bg_thread, bg_pool, bg_ctx = None, None, None
        if comp_pipelined and comp_futures:
            inv_perm_0 = np.argsort(perm)
            bg_pool = ThreadPoolExecutor(max_workers=max(MIN_WORKERS, (os.cpu_count() or 8) // CPU_WORKER_RATIO))
            try:
                from threadpoolctl import threadpool_limits
                bg_ctx = threadpool_limits(limits=1, user_api="blas")
            except ImportError:
                bg_ctx = nullcontext()

            def _bg_pipeline():
                from concurrent.futures import as_completed
                try:
                    idx_map = {id(f): i for i, f in enumerate(comp_futures)}
                    for f in as_completed(comp_futures, timeout=WORKER_TIMEOUT):
                        idx = idx_map[id(f)]
                        cr, seed_r = comp_seeds[idx]
                        lbl_r, cen_r, cnt_r, kd_r, ki_r, C_r, perm_r, V_diff_r, kd_vd_r, ki_vd_r, d_int_r = f.result(timeout=WORKER_TIMEOUT)
                        _kd = kd_vd_r if V_diff_r is not None else kd_r
                        _ki = ki_vd_r if V_diff_r is not None else ki_r
                        ks = [k for k in build_theoretical_k_sweep(C_r, n_samples=n, d_int=d_int_r) if k <= k_max_extra]
                        if not ks:
                            ks = [min(C_r - 1, k_max_extra)]
                        rep = {"cr": cr, "seed": seed_r, "labels": lbl_r, "centers": cen_r,
                               "counts": cnt_r, "knn_d": _kd, "knn_i": _ki, "C": C_r,
                               "perm": perm_r, "V_diff": V_diff_r, "k_sweep": ks, "k_max": k_max_extra, "d_int": d_int_r}
                        ri = len(bg_reps)
                        bg_reps.append(rep)
                        for k in ks:
                            bg_futs.append((bg_pool.submit(
                                scout_one_k, cen_r, _kd[:, :k], _ki[:, :k], cnt_r,
                                k, K_target, seed_r, V_diff_r, _kd, _ki, PEAK_DIVERSITY), ri, k))
                except Exception as e:
                    warnings.warn(f"Background pipeline error: {e}", RuntimeWarning)

            bg_ctx.__enter__()
            bg_thread = threading.Thread(target=_bg_pipeline, daemon=True)
            bg_thread.start()

        n_workers = min(len(k_sweep), max(MIN_WORKERS, (os.cpu_count() or 8) // CPU_WORKER_RATIO))
        if n_workers > 1 and C < C_SINGLE_THREAD:
            n_workers = 1

        def _run_k(k):
            return scout_one_k(self.leaf_centers, knn_dists[:, :k], knn_indices[:, :k],
                               self.leaf_counts, k, K_target, self.random_state,
                               V_diff, knn_dists, knn_indices)

        if n_workers <= 1 or len(k_sweep) <= 1:
            scout_results = [_run_k(k) for k in k_sweep]
        else:
            try:
                ctx = __import__('threadpoolctl').threadpool_limits(limits=1, user_api="blas")
            except ImportError:
                ctx = nullcontext()
            with ctx, ThreadPoolExecutor(max_workers=n_workers) as pool:
                scout_results = [f.result() for f in [pool.submit(_run_k, k) for k in k_sweep]]

        self.per_k_rows = []
        for res in sorted(scout_results, key=lambda r: r["k"]):
            k_val, K_det = res["k"], res["K_det_single"]
            if y is not None and res.get("best_lc") is not None:
                sl = res["best_lc"][self.sample_compact_leaf_ids]
                res["best_mets"] = {"ari": float(adjusted_rand_score(y, sl)),
                                    "nmi": float(normalized_mutual_info_score(y, sl)),
                                    "acc": float(clustering_accuracy(y, sl))}
                met_str = f" ARI={res['best_mets']['ari']:.4f} NMI={res['best_mets']['nmi']:.4f} ACC={res['best_mets']['acc']:.4f}"
            else:
                met_str = ""
            print(f"  k={k_val:>3d} | K_det={K_det:>3d} | time={res['elapsed']:.2f}s{met_str}")

        valid_scout = [r for r in scout_results if r.get("best_lc") is not None and int(r["best_lc"].max()) + 1 > 1]
        promoted_ks = [r["k"] for r in valid_scout]
        score_by_k = {r["k"]: 1.0 for r in valid_scout}

        seen_ks = set()
        for res in sorted(scout_results, key=lambda r: r["k"]):
            k_val = res["k"]
            if k_val in seen_ks:
                continue
            seen_ks.add(k_val)
            best_mets = res.get("best_mets", {"ari": np.nan, "nmi": np.nan, "acc": np.nan})
            self.per_k_rows.append({
                "dataset": name, "k": int(k_val), "method": "Ref+Eig",
                "K_det": int(res["K_det_single"]),
                "K_use": int(K_target) if K_target else int(res["K_det_single"]),
                "ari": best_mets["ari"], "nmi": best_mets["nmi"], "acc": best_mets["acc"],
                "ari_det": best_mets["ari"], "nmi_det": best_mets["nmi"], "acc_det": best_mets["acc"],
                "members": int(res["n_members"]), "n_proto": int(C),
                "labels": res["best_lc"][self.sample_compact_leaf_ids],
                "_best_lc": res["best_lc"], "_elapsed": res["elapsed"],
            })
            if self.best_k is None:
                self.best_k = k_val

        if bg_thread is not None:
            bg_thread.join(timeout=WORKER_TIMEOUT)
            cr_order = sorted(range(len(bg_reps)), key=lambda i: bg_reps[i]["cr"])
            old2new = {old: new for new, old in enumerate(cr_order)}
            extra_reps = [bg_reps[i] for i in cr_order]
            scout_x_map = {}
            for fut, old_ri, k in bg_futs:
                res = fut.result(timeout=WORKER_TIMEOUT) if hasattr(fut, 'result') else fut
                scout_x_map.setdefault(old2new[old_ri], []).append((k, res))
            if bg_pool:
                bg_pool.shutdown(wait=False)
            if bg_ctx:
                try:
                    bg_ctx.__exit__(None, None, None)
                except Exception:
                    pass
        else:
            extra_reps, scout_x_map = [], {}

        scout_by_rep = [[r for _, r in sorted(scout_x_map.get(ri, []), key=lambda x: x[0])]
                        for ri in range(len(extra_reps))]
        promoted_by_rep = []
        for ri in range(len(extra_reps)):
            valid = [r for r in scout_by_rep[ri] if r.get("best_lc") is not None and int(r["best_lc"].max()) + 1 > 1]
            promoted_by_rep.append(([r["k"] for r in valid], {r["k"]: 1.0 for r in valid}))

        best_anchor, best_anchor_C = -1, C
        for ri, rep in enumerate(extra_reps):
            if rep["C"] > best_anchor_C:
                best_anchor, best_anchor_C = ri, rep["C"]

        inv_perm_0 = np.argsort(perm)
        mc0_all = self.sample_compact_leaf_ids[inv_perm_0]
        if best_anchor == -1:
            N_ANCHOR, mc_anchor_all = C, mc0_all
            anchor_prom, anchor_sbk = list(promoted_ks), score_by_k
            anchor_smap = {r["k"]: r for r in scout_results if r.get("best_lc") is not None}
        else:
            arep = extra_reps[best_anchor]
            N_ANCHOR = arep["C"]
            mc_anchor_all = arep["labels"][np.argsort(arep["perm"])]
            anchor_prom, anchor_sbk = promoted_by_rep[best_anchor]
            anchor_smap = {r["k"]: r for r in scout_by_rep[best_anchor]}

        all_members, all_weights = [], []

        def _project(lc, s2mc):
            if s2mc is None:
                return lc.astype(np.int32)
            sl = lc[s2mc]
            vt = np.zeros((N_ANCHOR, int(sl.max()) + 1), dtype=np.int32)
            np.add.at(vt, (mc_anchor_all, sl), 1)
            return vt.argmax(axis=1).astype(np.int32)

        def _collect(prom, sbk, smap, s2mc):
            for kv in prom:
                sr = smap.get(kv)
                if sr is None:
                    continue
                lc = sr.get("best_lc")
                if lc is None or int(lc.max()) + 1 <= 1:
                    continue
                all_members.append(_project(lc, s2mc))
                all_weights.append(sbk.get(kv, 1.0))

        _collect(anchor_prom, anchor_sbk, anchor_smap, None)
        if best_anchor != -1:
            _collect(list(promoted_ks), score_by_k, {r["k"]: r for r in scout_results}, mc0_all)
        for ri, rep in enumerate(extra_reps):
            if ri == best_anchor:
                continue
            pr, sr = promoted_by_rep[ri]
            _collect(pr, sr, {r["k"]: r for r in scout_by_rep[ri]}, rep["labels"][np.argsort(rep["perm"])])

        w = np.array(all_weights, dtype=np.float64)
        if len(w) > 0:
            w = np.maximum(w, w.max() / len(w))
            w /= w.sum()
            mc_labels = _spectral_coassoc(all_members, N_ANCHOR, K_target, member_weights=w, random_state=self.random_state)
            sample_labels = mc_labels[mc_anchor_all][perm]
            K_det = int(sample_labels.max()) + 1
            if y is not None:
                ari_val = adjusted_rand_score(y, sample_labels)
                nmi_val = normalized_mutual_info_score(y, sample_labels)
                acc_val = clustering_accuracy(y, sample_labels)
            else:
                ari_val = nmi_val = acc_val = 0.0
            self.pooled_row = {
                "dataset": name, "k": "multi-seed", "method": f"anchor-consensus({len(w)})",
                "K_det": K_det, "K_use": K_target,
                "ari": ari_val, "nmi": nmi_val, "acc": acc_val,
                "ari_det": ari_val, "nmi_det": nmi_val, "acc_det": acc_val,
                "members": len(w), "n_proto": N_ANCHOR, "labels": sample_labels,
            }

        if self.per_k_rows:
            self.summary_best = (max(self.per_k_rows, key=lambda r: r.get("ari", float("-inf")))
                                 if y is not None else self.per_k_rows[0])
        self.t_total = time.time() - t0
        inv_perm = np.argsort(perm)
        for rd in [self.pooled_row, self.summary_best]:
            if rd and "labels" in rd and rd["labels"] is not None and len(rd["labels"]) == n:
                rd["labels"] = rd["labels"][inv_perm]
        if y is not None and self.summary_best:
            s = self.summary_best
            print(f"best single-k:   k={s['k']} | {s['method']} | K_det={s['K_det']} ARI_det={s['ari_det']:.4f} | ARI={s['ari']:.4f} NMI={s['nmi']:.4f} ACC={s['acc']:.4f}")
        if self.pooled_row:
            p = self.pooled_row
            print(f"pooled multi-k:  {p['method']} | K_det={p['K_det']} ARI_det={p['ari_det']:.4f} | ARI={p['ari']:.4f} NMI={p['nmi']:.4f} ACC={p['acc']:.4f}")
        return self


if __name__ == "__main__":
    from evogc_data import DATASET_REGISTRY, load_dataset
    SEED = 0
    DATASETS = ["mnist", "iris", "wine", "sipu_spiral", "sipu_jain", "sipu_D31",
        "sipu_R15", "sipu_a1", "sipu_s3", "sipu_s4", "sipu_aggregation",
        "sipu_compound", "sipu_flame", "sipu_path_based", "sipu_unbalance"]
    results = []
    for ds_key in DATASETS:
        if ds_key not in DATASET_REGISTRY:
            continue
        cfg = DATASET_REGISTRY[ds_key]
        X, y = load_dataset(cfg, random_state=SEED)
        t0 = time.time()
        model = EvoGC(K=cfg.true_k, random_state=SEED)
        model.fit(X, y=y, name=cfg.name)
        elapsed = time.time() - t0
        row = model.pooled_row or model.summary_best
        ari = row["ari"] if row else 0.0
        acc = row["acc"] if row else 0.0
        results.append((cfg.name, len(X), X.shape[1], cfg.true_k, ari, acc, elapsed))
    for name, n, d, K, ari, acc, t in results:
        print(f"{name:<16} {n:>6} {d:>4} {K:>3}   {ari:>6.3f} {acc:>6.3f} {t:>5.1f}s")
