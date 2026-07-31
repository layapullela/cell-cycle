"""
Merge per-phase HiCCUPS merged_loops.bedpe files into a single
consensus loop list, following the procedure:

  1. Read merged_loops for each phase × resolution (10 kb and 25 kb).
  2. For each pixel: q_value = max(fdrBL, fdrDonut).
  3. best_q  = min q_value across phases that contain this pixel.
     called_count = number of phases where the pixel appears.
  4. Keep pixels with best_q <= 0.01.
  5. Sort by best_q ascending (most significant first).
  6. Greedy collapse within 30 kb radius:
       - pick most-significant remaining pixel as target
       - cluster = all remaining pixels whose centre is within 30 kb of target
       - if cluster > 1 pixel: expand by cluster half-width and absorb extras
       - assign loop_id; repeat until no pixels remain.
  7. Summarise each cluster:
       loop_border  = bounding box of all pixels in cluster
       summit       = pixel with lowest best_q
       pixel_collapsed = number of pixels in cluster
       called_counts   = sum of called_count across cluster pixels

Output: raw_data/kang/hiccups/merged_loops_10kb.bedpe
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
HICCUPS_DIR = Path("/nfs/turbo/umms-minjilab/lpullela/cell-cycle/raw_data/kang/hiccups")
PHASES      = ["prometa", "anatelo", "earlyG1", "midG12", "lateG1"]
RESOLUTION  = 10_000          # bp
COLLAPSE_RADIUS = 30_000      # bp
Q_THRESH    = 0.01

OUT_FILE = HICCUPS_DIR / "merged_loops_10kb.bedpe"

# ── 1. read per-phase × resolution merged loops ───────────────────────────────
BEDPE_COLS = ["chr1","x1","x2","chr2","y1","y2",
              "name","score","strand1","strand2","color",
              "observed",
              "expectedBL","expectedDonut","expectedH","expectedV",
              "fdrBL","fdrDonut","fdrH","fdrV"]

def bedpe_paths():
    """Yield (phase_label, path) for every phase × resolution bedpe."""
    for phase in PHASES:
        yield phase, HICCUPS_DIR / phase / "merged_loops.bedpe"
        yield f"{phase}_25kbp", (
            HICCUPS_DIR / "25kbp" / f"{phase}_25kbp.hiccups" / "merged_loops.bedpe"
        )

def load_bedpe(phase: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#",
                     header=None, usecols=range(len(BEDPE_COLS)),
                     names=BEDPE_COLS)
    df["phase"] = phase
    return df

frames = []
for phase, path in bedpe_paths():
    if not path.is_file():
        print(f"Skipping (not found): {path}")
        continue
    frames.append(load_bedpe(phase, path))

all_pixels = pd.concat(frames, ignore_index=True)

# ── 2. per-pixel q_value ───────────────────────────────────────────────────────
all_pixels["q_value"] = all_pixels[["fdrBL", "fdrDonut"]].max(axis=1)

# pixel key: (chr1, x1, chr2, y1)  — bins uniquely identify the pixel
all_pixels["key"] = list(zip(all_pixels.chr1, all_pixels.x1,
                              all_pixels.chr2, all_pixels.y1))

# ── 3. best_q and called_count across phases ───────────────────────────────────
pixel_stats = (all_pixels
               .groupby("key")
               .agg(best_q=("q_value", "min"),
                    called_count=("phase", "count"))
               .reset_index())

# attach stats back to one representative row per pixel
rep = (all_pixels
       .sort_values("q_value")
       .drop_duplicates("key")        # keep row with lowest q per pixel
       .merge(pixel_stats, on="key"))

# ── 4. filter ─────────────────────────────────────────────────────────────────
rep = rep[rep["best_q"] <= Q_THRESH].copy()
print(f"Pixels passing q <= {Q_THRESH}: {len(rep)}")

# ── 5. sort by best_q ─────────────────────────────────────────────────────────
rep = rep.sort_values("best_q").reset_index(drop=True)

# pixel centres
rep["cx"] = (rep["x1"] + rep["x2"]) / 2
rep["cy"] = (rep["y1"] + rep["y2"]) / 2

# ── 6. greedy collapse ────────────────────────────────────────────────────────
remaining = rep.copy()
clusters  = []                      # list of DataFrames, one per loop_id

while len(remaining) > 0:
    target   = remaining.iloc[0]
    tcx, tcy = target["cx"], target["cy"]
    radius   = COLLAPSE_RADIUS

    # pixels within radius of target centre
    dist = np.sqrt((remaining["cx"] - tcx)**2 + (remaining["cy"] - tcy)**2)
    in_cluster = dist <= radius
    cluster_df = remaining[in_cluster].copy()

    if len(cluster_df) > 1:
        # compute bounding box and expand radius by cluster half-width
        dx = (cluster_df["x2"].max() - cluster_df["x1"].min()) / 2
        dy = (cluster_df["y2"].max() - cluster_df["y1"].min()) / 2
        half_width = max(dx, dy)
        expanded   = radius + half_width

        # absorb additional pixels now within expanded radius
        dist2      = np.sqrt((remaining["cx"] - tcx)**2 + (remaining["cy"] - tcy)**2)
        in_cluster = dist2 <= expanded
        cluster_df = remaining[in_cluster].copy()

    remaining = remaining[~in_cluster].reset_index(drop=True)
    clusters.append(cluster_df)

print(f"Clusters (loops) after collapse: {len(clusters)}")

# ── 7. summarise each cluster ─────────────────────────────────────────────────
rows = []
for loop_id, cluster in enumerate(clusters):
    summit = cluster.loc[cluster["best_q"].idxmin()]

    row = {
        # loop border = bounding box of all pixels
        "chr1":            summit["chr1"],
        "x1":              int(cluster["x1"].min()),
        "x2":              int(cluster["x2"].max()),
        "chr2":            summit["chr2"],
        "y1":              int(cluster["y1"].min()),
        "y2":              int(cluster["y2"].max()),
        # summit info
        "loop_id":         loop_id,
        "best_q":          summit["best_q"],
        "summit_x1":       int(summit["x1"]),
        "summit_x2":       int(summit["x2"]),
        "summit_y1":       int(summit["y1"]),
        "summit_y2":       int(summit["y2"]),
        "summit_phase":    summit["phase"],
        # collapse stats
        "pixel_collapsed": len(cluster),
        "called_counts":   int(cluster["called_count"].sum()),
    }
    rows.append(row)

result = pd.DataFrame(rows)
result = result.sort_values("best_q").reset_index(drop=True)

result.to_csv(OUT_FILE, sep="\t", index=False)
print(f"Wrote {len(result)} merged loops → {OUT_FILE}")
