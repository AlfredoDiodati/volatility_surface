import polars as pl
import numpy as np
from pathlib import Path

subfolder = "SPX"
full_path = f"data/{subfolder}/put/full.parquet"
bucket_path = f"data/{subfolder}/put/bucket.parquet"
output_file = f"test/bucket_stability_analysis.txt"

Path(output_file).parent.mkdir(parents=True, exist_ok=True)

full_df = pl.read_parquet(full_path)
bucket_df = pl.read_parquet(bucket_path)

delta_midpoints = {
    "mon1": -0.0625,
    "mon2": -0.25,
    "mon3": -0.4375,
    "mon4": -0.75,
}

maturity_midpoints = {
    "mat1": 26.0,
    "mat2": 67.5,
    "mat3": 135.0,
    "mat4": 270.0,
}

bucket_cols = [col for col in full_df.columns if col.startswith("bucket_")]
maturity_buckets = ["mat1", "mat2", "mat3", "mat4"]
moneyness_buckets = ["mon1", "mon2", "mon3", "mon4"]

bucket_stats = {}

for mat_name in maturity_buckets:
    for mon_name in moneyness_buckets:
        bucket_col = f"bucket_{mat_name}_{mon_name}"
        if bucket_col not in bucket_cols:
            continue
        
        candidates = full_df.filter(pl.col(bucket_col) == True)
        selected = bucket_df.filter(pl.col(bucket_col) == True)
        
        n_candidates = len(candidates)
        n_selected = len(selected)
        
        if n_selected == 0:
            continue
        
        selected_moneyness = selected["moneyness"].to_numpy()
        candidates_moneyness = candidates["moneyness"].to_numpy()
        
        selected_delta_dist = np.abs(selected_moneyness - delta_midpoints.get(mon_name, 0))
        candidates_delta_dist = np.abs(candidates_moneyness - delta_midpoints.get(mon_name, 0))
        
        bucket_stats[bucket_col] = {
            "n_selected": n_selected,
            "n_candidates": n_candidates,
            "n_days": selected["DATE"].n_unique(),
            "selected_mon_mean": float(np.mean(selected_moneyness)),
            "selected_mon_std": float(np.std(selected_moneyness)),
            "selected_mon_min": float(np.min(selected_moneyness)),
            "selected_mon_max": float(np.max(selected_moneyness)),
            "candidates_mon_mean": float(np.mean(candidates_moneyness)),
            "candidates_mon_std": float(np.std(candidates_moneyness)),
            "selected_delta_dist_mean": float(np.mean(selected_delta_dist)),
            "candidates_delta_dist_mean": float(np.mean(candidates_delta_dist)),
        }

output_lines = []
output_lines.append("=" * 80)
output_lines.append("BUCKETING STABILITY ANALYSIS")
output_lines.append("=" * 80)

for mat_name in maturity_buckets:
    output_lines.append(f"\n{mat_name.upper()}:")
    output_lines.append("-" * 80)
    for mon_name in moneyness_buckets:
        bucket_col = f"bucket_{mat_name}_{mon_name}"
        if bucket_col not in bucket_stats:
            continue
        stats = bucket_stats[bucket_col]
        n_selected = stats["n_selected"]
        n_candidates = stats["n_candidates"]
        n_days = stats["n_days"]
        pool_ratio = n_candidates / max(1, n_selected)
        
        output_lines.append(f"\n  {mon_name}:")
        output_lines.append(f"    Observations selected: {n_selected}")
        output_lines.append(f"    Days with >= 1 observation: {n_days}")
        output_lines.append(f"    Candidate pool size: {n_candidates} ({pool_ratio:.0f}x selected)")
        output_lines.append(f"    Selected moneyness: {stats['selected_mon_mean']:.6f} +/- {stats['selected_mon_std']:.6f}")
        output_lines.append(f"      Range: [{stats['selected_mon_min']:.6f}, {stats['selected_mon_max']:.6f}]")
        output_lines.append(f"    Candidate pool moneyness: {stats['candidates_mon_mean']:.6f} +/- {stats['candidates_mon_std']:.6f}")
        output_lines.append(f"    Avg distance to delta midpoint: selected={stats['selected_delta_dist_mean']:.6f}, candidates={stats['candidates_delta_dist_mean']:.6f}")

output_lines.append("\n" + "=" * 80)
output_lines.append("KEY FINDINGS")
output_lines.append("=" * 80)

mon1_stats = {k: v for k, v in bucket_stats.items() if "mon1" in k}
mon4_stats = {k: v for k, v in bucket_stats.items() if "mon4" in k}

if mon1_stats and mon4_stats:
    mon1_pool_ratio = np.mean([v["n_candidates"]/max(1, v["n_selected"]) for v in mon1_stats.values()])
    mon4_pool_ratio = np.mean([v["n_candidates"]/max(1, v["n_selected"]) for v in mon4_stats.values()])
    mon1_scatter = np.mean([v["selected_mon_std"] for v in mon1_stats.values()])
    mon4_scatter = np.mean([v["selected_mon_std"] for v in mon4_stats.values()])
    mon1_delta_dist = np.mean([v["selected_delta_dist_mean"] for v in mon1_stats.values()])
    mon4_delta_dist = np.mean([v["selected_delta_dist_mean"] for v in mon4_stats.values()])
    
    output_lines.append(f"\nmon1 (ATM, delta_midpoint=-0.0625):")
    output_lines.append(f"  Avg candidate pool ratio: {mon1_pool_ratio:.1f}x")
    output_lines.append(f"  Avg moneyness scatter: {mon1_scatter:.6f}")
    output_lines.append(f"  Avg distance from delta midpoint: {mon1_delta_dist:.6f}")
    
    output_lines.append(f"\nmon4 (deep OTM, delta_midpoint=-0.75):")
    output_lines.append(f"  Avg candidate pool ratio: {mon4_pool_ratio:.1f}x")
    output_lines.append(f"  Avg moneyness scatter: {mon4_scatter:.6f}")
    output_lines.append(f"  Avg distance from delta midpoint: {mon4_delta_dist:.6f}")
    
    output_lines.append(f"\nStability comparison:")
    output_lines.append(f"  Pool ratio (mon1/mon4): {mon1_pool_ratio/max(1e-6, mon4_pool_ratio):.2f}x")
    output_lines.append(f"  Scatter (mon1/mon4): {mon1_scatter/max(1e-6, mon4_scatter):.2f}x")
    output_lines.append(f"  Delta distance (mon1/mon4): {mon1_delta_dist/max(1e-6, mon4_delta_dist):.2f}x")
    
    if mon1_scatter > 2 * mon4_scatter:
        output_lines.append(f"\nmon1 is UNSTABLE: >2x scatter of mon4 - bucketing artifact likely")
    elif mon1_pool_ratio < mon4_pool_ratio / 5:
        output_lines.append(f"\nmon1 has THIN POOL: <1/5 the candidate density of mon4")
    elif mon1_delta_dist > 2 * mon4_delta_dist:
        output_lines.append(f"\nmon1 DIVERGES from target: selected options are 2x farther from delta midpoint")
    else:
        output_lines.append(f"\nmon1 appears stable relative to mon4")

with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))