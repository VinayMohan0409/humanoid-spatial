"""Change region stage: fuses geometric and appearance evidence into one change flag per point, removes speckle with a majority filter, and clusters the flagged points into change regions for each video."""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from appearance import appearance_maps
from changes import region_colors
from compare import aligned_points, see_through_status
from config import CHECK_FRAME, SMOOTH_K, Z_STRONG, Z_WEAK
from point_clouds import make_cloud
from segment import grow_objects, render_check

EVIDENCE_COLORS = {"unchanged": (0.55, 0.55, 0.55), "unknown": (0.25, 0.25, 0.25), "geometric": (0.95, 0.40, 0.40), "appearance": (0.98, 0.70, 0.15)}


def robust_z(values, mask):
    """Standardise values by the median and MAD (scaled to a standard deviation) of the masked ones."""
    median = np.median(values[mask])
    return (values - median) / max(1.4826 * np.median(np.abs(values[mask] - median)), 1e-6)


def change_evidence(video, own_in_other, other, other_in_own):
    """Return per-point (geometric, appearance, matched, feature z-score): geometric where the other video's cameras saw empty space at the point; appearance where the other video has surface there but the feature differs strongly, or moderately together with chroma."""
    matched, dissimilarity, chroma = appearance_maps(video, other_in_own, other)
    geometric = ~matched & (see_through_status(own_in_other, other) == -1)
    z_feature, z_chroma = robust_z(dissimilarity, matched), robust_z(chroma, matched)
    appearance = matched & ((z_feature >= Z_STRONG) | ((z_feature >= Z_WEAK) & (z_chroma >= Z_WEAK)))
    return geometric, appearance, matched, z_feature


def majority_filter(flag, points):
    """Keep a flagged point only if at least half of its SMOOTH_K nearest neighbours (itself included) are flagged."""
    neighbours = KDTree(points).query(points, k=SMOOTH_K)[1]
    return flag & (flag[neighbours].mean(axis=1) >= 0.5)


def change_regions(video, own_in_other, other, other_in_own):
    """Return a dict of per-point arrays: region labels (feature-gated clusters of the smoothed change flags), changed and geometric flags, matched flag and the feature z-score."""
    geometric, appearance, matched, z_feature = change_evidence(video, own_in_other, other, other_in_own)
    changed = majority_filter(geometric | appearance, video["points"])
    labels = np.zeros(len(changed), dtype=int)
    labels[changed] = grow_objects(video["points"][changed], video["features"][changed])
    return {"labels": labels, "changed": changed, "geometric": geometric, "matched": matched, "z_feature": z_feature}


def evidence_colors(evidence):
    """Colour points grey (unchanged), dark grey (unknown), orange (appearance change) or red (geometric change)."""
    colors = np.tile(EVIDENCE_COLORS["unchanged"], (len(evidence["changed"]), 1))
    colors[~evidence["matched"] & ~evidence["geometric"]] = EVIDENCE_COLORS["unknown"]
    colors[evidence["changed"]] = EVIDENCE_COLORS["appearance"]
    colors[evidence["changed"] & evidence["geometric"]] = EVIDENCE_COLORS["geometric"]
    return colors


def export_change_regions(before, after, out_dir):
    """Write per-video evidence clouds (red geometric, orange appearance) and region clouds with check-frame images, and print each region."""
    after_in_before, before_in_after = aligned_points(before, after)
    runs = {"before": (before, before_in_after, after, after_in_before), "after": (after, after_in_before, before, before_in_after)}
    files = {"evidence": {}, "regions": {}}
    for name, args in runs.items():
        evidence = change_regions(*args)
        labels = evidence["labels"]
        regions = np.where(labels[:, None] > 0, region_colors(labels), EVIDENCE_COLORS["unchanged"])
        for kind, colors in (("evidence", evidence_colors(evidence)), ("regions", regions)):
            files[kind][name] = out_dir / f"{name}_change_{kind}.ply"
            o3d.io.write_point_cloud(str(files[kind][name]), make_cloud(args[0]["points"], colors))
        print(f"\n{name}: {labels.max()} change regions")
        for k in range(1, labels.max() + 1):
            members = labels == k
            print(f"  region {k:2d}: {members.sum():5d} pts, {evidence['geometric'][members].mean():4.0%} geometric, centre {np.round(args[0]['points'][members].mean(axis=0), 2)}")
    for kind in files:
        render_check(files[kind], CHECK_FRAME, out_dir / f"change_{kind}_frame{CHECK_FRAME}.jpg")
