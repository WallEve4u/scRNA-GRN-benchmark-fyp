import json
from pathlib import Path
import numpy as np
import pandas as pd


CONFIG = {
    "datasets": {
        "hESC-500": {"tf_file": "data/prepared/human-tfs.txt"},
        "mHSC-E-500": {"tf_file": "data/prepared/mouse-tfs.txt"},
        "mDC-500": {"tf_file": "data/prepared/mouse-tfs.txt"},
    },
    "stress_axes": {
        "dropout": [0.1, 0.3, 0.5],
        "noise": [0.1, 0.25],
        "subsample": [100, 250, 500],
    },
    "replicates": [1, 2, 3],
    "base_seed": 2026,
    "gate1_cut_list": {
        "genie3_replicates": [1, 2],
        "genie3_mDC_axes": ["baseline", "dropout"],
        "grnboost2_replicates": [1, 2, 3],
        "ppcor_replicates": [1, 2, 3],
        "local_n_jobs": 4,
    },
}


def make_seed(ds_idx, axis_idx, level_idx, rep):
    """Deterministic unique integer seed for every (dataset, axis, level, replicate)."""
    return CONFIG["base_seed"] + ds_idx * 1000 + axis_idx * 100 + level_idx * 10 + rep


def inject_dropout(df, rate, seed):
    """Mask `rate` fraction of existing NON-ZERO entries to 0.0."""
    rng = np.random.default_rng(seed)
    vals = df.values.copy()
    nz_mask = vals > 0
    keep = rng.binomial(1, 1.0 - rate, size=int(nz_mask.sum()))
    vals[nz_mask] = vals[nz_mask] * keep
    return pd.DataFrame(vals, index=df.index, columns=df.columns)


def inject_noise(df, level, seed):
    """Add zero-mean Gaussian noise scaled to `level * gene_SD`, clipped at 0."""
    rng = np.random.default_rng(seed)
    vals = df.values.copy()
    stds = vals.std(axis=1, keepdims=True)
    noise = rng.normal(loc=0.0, scale=level * stds, size=vals.shape)
    # Leave zero-SD genes untouched and clip log-scale values at 0
    noisy = np.where(stds > 0, np.clip(vals + noise, 0.0, None), vals)
    return pd.DataFrame(noisy, index=df.index, columns=df.columns)


def subsample_cells(df, n_cells, seed):
    """Draw `n_cells` columns without replacement, preserving original column order."""
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(df.shape[1], size=n_cells, replace=False))
    return df.iloc[:, chosen]


def main():
    # 1. Save frozen config.json
    Path("config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    print("Saved frozen config.json")

    skipped_rows = []
    manifest_rows = []
    axis_order = ["dropout", "noise", "subsample"]

    for ds_idx, ds in enumerate(CONFIG["datasets"].keys()):
        expr = pd.read_csv(f"data/prepared/{ds}-ExpressionData.csv", index_col=0)
        out_dir = Path(f"work/matrices/{ds}")
        out_dir.mkdir(parents=True, exist_ok=True)
        n_genes, n_cells = expr.shape
        base_nz = int((expr.values > 0).sum())
        print(f"\nGenerating stress matrices for {ds} ({n_genes} genes x {n_cells} cells)...")

        for axis_idx, axis in enumerate(axis_order):
            for level_idx, level in enumerate(CONFIG["stress_axes"][axis]):
                if axis == "subsample" and level >= n_cells:
                    skipped_rows.append({
                        "dataset": ds, "axis": axis, "level": level,
                        "reason": f"subsample level {level} >= dataset cell count ({n_cells})"
                    })
                    print(f"  [SKIP] {ds} subsample_{level} (>= {n_cells} cells)")
                    continue

                for rep in CONFIG["replicates"]:
                    seed = make_seed(ds_idx, axis_idx, level_idx, rep)
                    if axis == "dropout":
                        stressed = inject_dropout(expr, level, seed)
                    elif axis == "noise":
                        stressed = inject_noise(expr, level, seed)
                    else:
                        stressed = subsample_cells(expr, level, seed)

                    fname = f"{axis}_{level}_r{rep}.csv"
                    fpath = out_dir / fname
                    stressed.to_csv(fpath)

                    # Quick sanity stats
                    nz_kept_pct = (stressed.values > 0).sum() / base_nz * 100
                    min_val = float(stressed.values.min())
                    manifest_rows.append({
                        "dataset": ds, "axis": axis, "level": level, "rep": rep,
                        "seed": seed, "path": str(fpath),
                        "n_cells": stressed.shape[1],
                        "nz_kept_pct": round(nz_kept_pct, 1),
                        "min_val": round(min_val, 4),
                    })

    pd.DataFrame(skipped_rows).to_csv("work/skipped.csv", index=False)
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv("work/matrices_manifest.csv", index=False)

    # Print sanity check summary
    print("\n=== SANITY CHECK SUMMARY ===")
    print(f"Total stress matrices created: {len(manifest_df)}")
    print(f"Total skipped levels logged  : {len(skipped_rows)} (see work/skipped.csv)")
    print(f"Any negative values created? : {(manifest_df['min_val'] < 0).any()}")
    drop30 = manifest_df[(manifest_df["axis"] == "dropout") & (manifest_df["level"] == 0.3)]
    print(f"Avg non-zeros kept at 30% dropout: {drop30['nz_kept_pct'].mean():.1f}% (expected ~70.0%)")
    print("Saved manifest to work/matrices_manifest.csv")


if __name__ == "__main__":
    main()