"""Diagnostics: for every processed scene, writes distance heatmaps, appearance maps, change regions and coloured moved/added/removed verdicts, and prints the matching decisions (usage: python pipeline/diagnose.py [SCENE_DIR ...])."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def find_scenes():
    """Return every scene folder under output/ that has already been segmented."""
    return [d for d in sorted((ROOT / "output").iterdir()) if (d / "segments").exists()]


def scene_voxel(scene):
    """Read the scene's voxel size back from its fused-cloud file name."""
    return re.search(r"vox([\d.]+)\.ply", next((scene / "tsdf").glob("before_*.ply")).name)[1]


def write_heatmaps(before, after, out_dir):
    """Colour each video's points by distance to the other video's cloud (own coordinates) and render them into the check frame."""
    import open3d as o3d
    from changes import compute_cloud_distances, create_distance_heatmap
    from compare import aligned_points
    from config import CHECK_FRAME
    from point_clouds import make_cloud
    from segment import render_check

    after_in_before, before_in_after = aligned_points(before, after)
    files = {}
    for name, own, other in (("before", before["points"], after_in_before), ("after", after["points"], before_in_after)):
        source = make_cloud(own)
        heatmap = create_distance_heatmap(source, compute_cloud_distances(source, make_cloud(other)))
        files[name] = out_dir / f"{name}_heatmap.ply"
        o3d.io.write_point_cloud(str(files[name]), heatmap)
    render_check(files, CHECK_FRAME, out_dir / f"heatmap_frame{CHECK_FRAME}.jpg")


def write_colour_clouds(before, after, out_dir):
    """Write both videos' true-colour clouds in before's coordinates, so they can be overlaid and inspected in 3D."""
    import open3d as o3d
    from compare import aligned_points
    from point_clouds import make_cloud

    after_in_before, _ = aligned_points(before, after)
    for name, points, video in (("before", before["points"], before), ("after", after_in_before, after)):
        o3d.io.write_point_cloud(str(out_dir / f"{name}_colour.ply"), make_cloud(points, video["colors"]))


def diagnose_scene():
    """Run the diagnostics for the scene selected by PIPELINE_OUTPUT / PIPELINE_VOXEL."""
    from appearance import export_appearance
    from change_regions import export_change_regions
    from match_changes import export_verdicts
    from compare import load_video
    from config import OUTPUT

    before, after = load_video("before"), load_video("after")
    out_dir = OUTPUT / "diagnose"
    out_dir.mkdir(exist_ok=True)
    write_colour_clouds(before, after, out_dir)
    write_heatmaps(before, after, out_dir)
    export_appearance(before, after, out_dir)
    export_change_regions(before, after, out_dir)
    export_verdicts(before, after, out_dir)


def main():
    """Spawn one process per scene (config reads the voxel size at import) and run its diagnostics."""
    parser = argparse.ArgumentParser(description="Heatmaps and per-object evidence for processed scenes.")
    parser.add_argument("scenes", nargs="*", type=Path, help="scene output folders (default: all processed scenes)")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        diagnose_scene()
        return
    for scene in [s.resolve() for s in args.scenes] or find_scenes():
        print(f"\n######## {scene.name}", flush=True)
        env = {**os.environ, "PIPELINE_OUTPUT": str(scene), "PIPELINE_VOXEL": scene_voxel(scene)}
        subprocess.run([sys.executable, __file__, "--child"], env=env, check=True)


if __name__ == "__main__":
    main()
