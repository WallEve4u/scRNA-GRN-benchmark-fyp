import pandas as pd
import numpy as np
from pathlib import Path

# 1.one-TF-per-line .txt files from BEELINE's TF CSVs
for species in ["human", "mouse"]:
    tf_csv = Path(f"data/beeline/{species}-tfs.csv")
    tf_df = pd.read_csv(tf_csv)
    tfs = tf_df.iloc[:, 0].dropna().astype(str).str.strip().unique()
    out_txt = Path(f"data/prepared/{species}-tfs.txt")
    out_txt.write_text("\n".join(tfs) + "\n", encoding="utf-8")

# 2. Define the 3 datasets and their species TF file
datasets = [
    ("hESC-500", "data/prepared/human-tfs.txt"),
    ("mHSC-E-500", "data/prepared/mouse-tfs.txt"),
    ("mDC-500", "data/prepared/mouse-tfs.txt"),
]

subsample_levels = [100, 250, 500]
report_lines = []
hard_fail = False

for name, tf_file in datasets:
    expr_path = Path(f"data/prepared/{name}-ExpressionData.csv")
    net_path = Path(f"data/prepared/{name}-network.csv")

    # Load expression matrix (rows = genes, cols = cells) and reference network
    expr = pd.read_csv(expr_path, index_col=0)
    net = pd.read_csv(net_path)
    tf_set = set(Path(tf_file).read_text(encoding="utf-8").splitlines())

    n_genes, n_cells = expr.shape
    n_nan = int(expr.isna().sum().sum())
    n_neg = int((expr.values < 0).sum())
    n_dup_genes = int(expr.index.duplicated().sum())
    max_val = float(expr.values.max())
    zero_frac = float((expr.values == 0).mean() * 100)
    low_nz_genes = int(((expr.values > 0).sum(axis=1) <= 3).sum())
    tfs_in_matrix = len(set(expr.index).intersection(tf_set))

    # Network checks
    net_cols = list(net.columns)
    self_loops = int((net["Gene1"] == net["Gene2"]).sum())
    dup_edges = int(net.duplicated(subset=["Gene1", "Gene2"]).sum())
    net_genes = set(net["Gene1"]).union(set(net["Gene2"]))
    missing_genes = len(net_genes - set(expr.index))
    ref_tfs = net["Gene1"].nunique()
    n_edges = len(net)

    # Check which subsample levels are valid (< n_cells)
    valid_subs = [s for s in subsample_levels if s < n_cells]
    skipped_subs = [s for s in subsample_levels if s >= n_cells]

    if n_nan > 0 or n_neg > 0 or n_dup_genes > 0 or self_loops > 0 or dup_edges > 0 or missing_genes > 0 or tfs_in_matrix == 0:
        hard_fail = True

    block = f"""
=== QC REPORT: {name} ===
Shape (Genes x Cells)   : {n_genes} x {n_cells}
NaNs / Negatives / Dups : {n_nan} / {n_neg} / {n_dup_genes}
Max expression value    : {max_val:.2f} (log-scaled if < 30)
Baseline zero fraction  : {zero_frac:.1f}%
Genes with <=3 non-zeros: {low_nz_genes}
Known TFs in matrix     : {tfs_in_matrix}
Network columns         : {net_cols}
True edges (k) / Ref TFs: {n_edges} edges / {ref_tfs} source TFs
Self-loops / Dup edges  : {self_loops} / {dup_edges}
Missing network genes   : {missing_genes}
Valid subsample levels  : {valid_subs}  (Skipped: {skipped_subs})
"""
    print(block)
    report_lines.append(block)

Path("data/prepared/qc_summary.txt").write_text("\n".join(report_lines), encoding="utf-8")
print("Saved QC report to data/prepared/qc_summary.txt")

if hard_fail:
    raise SystemExit("HARD QC FAILURE: check numbers above!")
else:
    print("ALL QC CHECKS PASSED!")