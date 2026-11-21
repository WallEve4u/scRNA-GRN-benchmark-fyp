import argparse
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score


class PeakRSSMonitor:
    """Measures peak RAM (MB) of the main Python process + all joblib worker processes."""
    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def _poll(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                for child in self._proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                self.peak_mb = max(self.peak_mb, rss / (1024 * 1024))
            except psutil.NoSuchProcess:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()


def run_ppcor(expr_df, tf_list):
    """Partial Spearman correlation via regularized precision matrix (pseudo-inverse)."""
    genes = list(expr_df.index)
    X = expr_df.values  # shape: (n_genes, n_cells)
    # Filter zero-variance genes so correlation never produces NaN
    stds = X.std(axis=1)
    keep = stds > 0
    X_keep = X[keep]
    genes_keep = [g for g, k in zip(genes, keep) if k]

    # Rank each gene across cells (Spearman) and compute correlation matrix
    ranks = np.apply_along_axis(rankdata, 1, X_keep)
    corr = np.corrcoef(ranks)
    prec = np.linalg.pinv(corr)
    diag = np.sqrt(np.maximum(np.diag(prec), 1e-12))
    pcorr = -prec / np.outer(diag, diag)
    np.fill_diagonal(pcorr, 0.0)

    tf_set = set(tf_list)
    rows = []
    for i, g1 in enumerate(genes_keep):
        if g1 not in tf_set:
            continue
        for j, g2 in enumerate(genes_keep):
            if i != j:
                rows.append((g1, g2, abs(float(pcorr[i, j]))))

    out = pd.DataFrame(rows, columns=["Gene1", "Gene2", "EdgeWeight"])
    return out.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)


def _fit_one_target(target_idx, X_tfs, y, tf_names, target_name, algo, seed):
    if np.std(y) == 0:
        return []
    # Exclude target gene itself if it is also a TF (no self-loops)
    mask = [i for i, tf in enumerate(tf_names) if tf != target_name]
    X_sub = X_tfs[:, mask]
    tfs_sub = [tf_names[i] for i in mask]

    if algo == "grnboost2":
        model = GradientBoostingRegressor(
            learning_rate=0.01,
            n_estimators=5000,
            max_features=0.1,
            subsample=0.9,
            n_iter_no_change=25,
            tol=1e-4,
            random_state=seed + target_idx,
        )
    else:  # genie3
        model = RandomForestRegressor(
            n_estimators=1000,
            max_features="sqrt",
            random_state=seed + target_idx,
            n_jobs=1,
        )
    model.fit(X_sub, y)
    return [(tf, target_name, float(w)) for tf, w in zip(tfs_sub, model.feature_importances_) if w > 0]


def run_tree_grn(expr_df, tf_list, algo, seed=42, n_jobs=4):
    """Dask-free GENIE3 or GRNBoost2 using joblib multiprocessing."""
    genes = list(expr_df.index)
    tf_names = [g for g in genes if g in set(tf_list)]
    X_tfs = expr_df.loc[tf_names].values.T  # shape: (n_cells, n_tfs)
    Y_all = expr_df.values  # shape: (n_genes, n_cells)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_target)(i, X_tfs, Y_all[i], tf_names, genes[i], algo, seed)
        for i in range(len(genes))
    )
    rows = [edge for sub in results for edge in sub]
    out = pd.DataFrame(rows, columns=["Gene1", "Gene2", "EdgeWeight"])
    return out.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)


def evaluate_edges(pred_df, ref_df):
    """BEELINE-style AUROC, AUPRC ratio, EP, and EPR."""
    ref_edges = set(zip(ref_df["Gene1"], ref_df["Gene2"]))
    ref_genes = sorted(set(ref_df["Gene1"]).union(set(ref_df["Gene2"])))
    ref_tfs = sorted(set(ref_df["Gene1"]))

    # Map predicted weights (absolute value)
    pred_map = dict(zip(zip(pred_df["Gene1"], pred_df["Gene2"]), pred_df["EdgeWeight"].abs()))

    # 1. AUROC & AUPRC over all ordered non-self pairs in reference gene universe
    y_true, y_score = [], []
    for g1 in ref_genes:
        for g2 in ref_genes:
            if g1 == g2:
                continue
            y_true.append(1 if (g1, g2) in ref_edges else 0)
            y_score.append(pred_map.get((g1, g2), 0.0))
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    density = y_true.mean()
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    auprc_ratio = auprc / density if density > 0 else 0.0

    # 2. Early Precision (EP) & Early Precision Ratio (EPR) over TF -> gene pairs
    cand_pairs = [(tf, g) for tf in ref_tfs for g in ref_genes if tf != g]
    cand_scores = [(pred_map.get(p, 0.0), 1 if p in ref_edges else 0) for p in cand_pairs]
    cand_scores.sort(key=lambda x: x[0], reverse=True)
    k = len(ref_edges)
    cutoff_score = cand_scores[min(k - 1, len(cand_scores) - 1)][0]
    top_k = [item for item in cand_scores if item[0] >= cutoff_score and item[0] > 0]
    ep = sum(x[1] for x in top_k) / len(top_k) if top_k else 0.0
    tf_density = k / len(cand_pairs)
    epr = ep / tf_density if tf_density > 0 else 0.0
    return auroc, auprc_ratio, ep, epr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hESC-500", "mHSC-E-500", "mDC-500"])
    parser.add_argument("--algos", nargs="+", default=["ppcor", "grnboost2"])
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf_files = {
        "hESC-500": "data/prepared/human-tfs.txt",
        "mHSC-E-500": "data/prepared/mouse-tfs.txt",
        "mDC-500": "data/prepared/mouse-tfs.txt",
    }
    Path("work/pilot").mkdir(parents=True, exist_ok=True)
    log_rows = []

    for ds in args.datasets:
        expr = pd.read_csv(f"data/prepared/{ds}-ExpressionData.csv", index_col=0)
        ref = pd.read_csv(f"data/prepared/{ds}-network.csv")[["Gene1", "Gene2"]]
        tfs = Path(tf_files[ds]).read_text(encoding="utf-8").splitlines()

        for algo in args.algos:
            print(f"\nRunning {algo.upper()} on {ds} (n_jobs={args.n_jobs})...")
            with PeakRSSMonitor() as mon:
                t0 = time.perf_counter()
                if algo == "ppcor":
                    edges = run_ppcor(expr, tfs)
                else:
                    edges = run_tree_grn(expr, tfs, algo=algo, seed=args.seed, n_jobs=args.n_jobs)
                elapsed = time.perf_counter() - t0

            out_file = f"work/pilot/{ds}_{algo}_baseline.tsv"
            edges.to_csv(out_file, sep="\t", index=False)
            auroc, auprc_ratio, ep, epr = evaluate_edges(edges, ref)
            print(
                f"  DONE in {elapsed:.1f}s | Peak RAM: {mon.peak_mb:.0f} MB | "
                f"AUROC={auroc:.3f} | AUPRC_ratio={auprc_ratio:.2f}x | EP={ep:.3f} | EPR={epr:.2f}x"
            )
            log_rows.append({
                "dataset": ds, "algo": algo, "seconds": round(elapsed, 2),
                "peak_rss_mb": round(mon.peak_mb, 1), "auroc": round(auroc, 4),
                "auprc_ratio": round(auprc_ratio, 3), "ep": round(ep, 4), "epr": round(epr, 3),
            })

    log_df = pd.DataFrame(log_rows)
    log_path = Path("work/pilot/pilot_log.csv")
    log_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(f"\nSaved pilot results to {log_path}")


if __name__ == "__main__":
    main()