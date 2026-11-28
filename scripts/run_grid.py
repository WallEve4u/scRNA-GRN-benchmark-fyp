import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# Allow importing directly from scripts/run_pilot.py on both Windows and Kaggle
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pilot import PeakRSSMonitor, evaluate_edges, run_ppcor, run_tree_grn


def build_job_list(config, manifest, selected_datasets, selected_algos):
    cut = config["gate1_cut_list"]
    jobs = []

    for ds in selected_datasets:
        ds_manifest = manifest[manifest["dataset"] == ds]
        base_expr_path = f"data/prepared/{ds}-ExpressionData.csv"

        for algo in selected_algos:
            if algo == "ppcor":
                base_reps = [1]  # PPCOR is deterministic on baseline
                stress_reps = cut["ppcor_replicates"]
                allowed_axes = ["baseline", "dropout", "noise", "subsample"]
            elif algo == "grnboost2":
                base_reps = cut["grnboost2_replicates"]
                stress_reps = cut["grnboost2_replicates"]
                allowed_axes = ["baseline", "dropout", "noise", "subsample"]
            else:  # genie3 (Gate 1 cut list)
                base_reps = cut["genie3_replicates"]
                stress_reps = cut["genie3_replicates"]
                allowed_axes = cut["genie3_mDC_axes"] if ds == "mDC-500" else ["baseline", "dropout", "noise", "subsample"]

            # 1. Baseline jobs
            for rep in base_reps:
                jobs.append({
                    "run_id": f"{ds}__{algo}__baseline__0__r{rep}",
                    "dataset": ds, "algo": algo, "axis": "baseline",
                    "level": 0, "rep": rep,
                    "seed": config["base_seed"] + rep,
                    "expr_path": base_expr_path,
                })

            # 2. Stress jobs from manifest
            for _, row in ds_manifest.iterrows():
                if row["axis"] not in allowed_axes:
                    continue
                if int(row["rep"]) not in stress_reps:
                    continue
                jobs.append({
                    "run_id": f"{ds}__{algo}__{row['axis']}__{row['level']}__r{int(row['rep'])}",
                    "dataset": ds, "algo": algo, "axis": row["axis"],
                    "level": row["level"], "rep": int(row["rep"]),
                    "seed": int(row["seed"]),
                    "expr_path": row["path"],
                })
    return jobs


def append_csv(path, row_dict):
    df = pd.DataFrame([row_dict])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hESC-500", "mHSC-E-500", "mDC-500"])
    parser.add_argument("--algos", nargs="+", default=["ppcor"])
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv("work/matrices_manifest.csv")
    jobs = build_job_list(config, manifest, args.datasets, args.algos)

    log_path = Path("work/runs_log.csv")
    metrics_path = Path("work/metrics.csv")

    done_ids = set()
    if log_path.exists():
        prev = pd.read_csv(log_path)
        done_ids = set(prev.loc[prev["status"] == "OK", "run_id"])

    print(f"Total planned jobs: {len(jobs)} | Already done: {len([j for j in jobs if j['run_id'] in done_ids])}")

    # Cache reference networks and TF lists in memory
    ref_cache = {}
    tf_cache = {}
    for ds in args.datasets:
        ref_cache[ds] = pd.read_csv(f"data/prepared/{ds}-network.csv")[["Gene1", "Gene2"]]
        tf_cache[ds] = Path(config["datasets"][ds]["tf_file"]).read_text(encoding="utf-8").splitlines()

    for idx, job in enumerate(jobs, start=1):
        rid = job["run_id"]
        if rid in done_ids:
            continue

        print(f"[{idx}/{len(jobs)}] Running {rid} ...")
        try:
            expr = pd.read_csv(job["expr_path"], index_col=0)
            out_dir = Path(f"work/edges/{job['dataset']}/{job['algo']}")
            out_dir.mkdir(parents=True, exist_ok=True)
            edge_file = out_dir / f"{rid}.tsv"

            with PeakRSSMonitor() as mon:
                t0 = time.perf_counter()
                if job["algo"] == "ppcor":
                    edges = run_ppcor(expr, tf_cache[job["dataset"]])
                else:
                    edges = run_tree_grn(
                        expr, tf_cache[job["dataset"]],
                        algo=job["algo"], seed=job["seed"], n_jobs=args.n_jobs
                    )
                elapsed = time.perf_counter() - t0

            edges.to_csv(edge_file, sep="\t", index=False)
            auroc, auprc_ratio, ep, epr = evaluate_edges(edges, ref_cache[job["dataset"]])

            append_csv(metrics_path, {
                "run_id": rid, "dataset": job["dataset"], "algo": job["algo"],
                "axis": job["axis"], "level": job["level"], "rep": job["rep"],
                "seconds": round(elapsed, 2), "peak_rss_mb": round(mon.peak_mb, 1),
                "auroc": round(auroc, 4), "auprc_ratio": round(auprc_ratio, 3),
                "ep": round(ep, 4), "epr": round(epr, 3),
            })
            append_csv(log_path, {
                "run_id": rid, "status": "OK",
                "seconds": round(elapsed, 2), "peak_rss_mb": round(mon.peak_mb, 1),
                "error": "",
            })
            print(f"    OK in {elapsed:.1f}s | RAM={mon.peak_mb:.0f} MB | AUPRC_ratio={auprc_ratio:.2f}x | EPR={epr:.2f}x")

        except Exception as e:
            err_tail = traceback.format_exc().splitlines()[-1]
            append_csv(log_path, {
                "run_id": rid, "status": "FAILED",
                "seconds": 0.0, "peak_rss_mb": 0.0, "error": err_tail,
            })
            print(f"    FAILED: {err_tail}")

    print("\nBatch finished! Check work/metrics.csv and work/runs_log.csv.")


if __name__ == "__main__":
    main()