"""Single entry point: given a before and an after video, runs every pipeline stage and writes the verdict clouds (changed objects tinted blue = moved, green = added, red = removed) and verdicts.json (usage: python pipeline/main.py BEFORE AFTER [--name NAME])."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VOXEL_FRACTION = 0.0065  # voxel size as a share of the scene's median depth (0.005 for our first scene)
STATUS_LEGEND = "true colours = unchanged, blue = moved, green = added, red = removed"


def parse_args():
    """Read the command line: the two videos and an optional run name."""
    parser = argparse.ArgumentParser(description="Detect added, removed and moved objects between two videos of a scene.")
    parser.add_argument("before", type=Path, help="video of the scene before the change")
    parser.add_argument("after", type=Path, help="video of the scene after the change")
    parser.add_argument("--name", help="name of the scene folder under output/ (default: derived from the video names)")
    parser.add_argument("--no-labels", action="store_true", help="skip naming the changed objects with the vision-language model")
    parser.add_argument("--phase", type=int, default=1, help=argparse.SUPPRESS)  # internal: 2 continues after the reconstruction
    return parser.parse_args()


def run_name(before, after):
    """Return a folder name for this pair: the shared parent folder if there is one, otherwise both file names."""
    if before.parent == after.parent and before.parent.name not in ("data", ""):
        return before.parent.name
    return f"{before.stem}_vs_{after.stem}"


def describe_videos(before, after):
    """Print duration, frame count and resolution of both input videos."""
    for label, path in (("before", before), ("after", after)):
        cap = cv2.VideoCapture(str(path))
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        print(f"{label:6s} {path.name}: {frames / fps:.1f} s, {int(frames)} frames, {int(cap.get(3))}x{int(cap.get(4))}")
        cap.release()


def scene_voxel(out_dir):
    """Choose the voxel size from the reconstruction: a fixed share of the median depth, kept within sensible bounds."""
    depths = [np.load(out_dir / "vggt" / f"{name}.npz")["depth"] for name in ("before", "after")]
    return float(np.clip(VOXEL_FRACTION * np.median(np.concatenate([d.ravel() for d in depths])), 0.003, 0.03))


def cleaned_path(out_dir, name):
    """Return where the cleaned copy of the before or after video is kept."""
    return out_dir / "input" / f"{name}.mp4"


def clean_inputs(before, after, out_dir):
    """Write cleaned copies of both videos to out_dir/input/ (the originals are left untouched)."""
    from clean_video import clean_video

    (out_dir / "input").mkdir(parents=True, exist_ok=True)
    for name, source in (("before", before), ("after", after)):
        clean_video(source, cleaned_path(out_dir, name))


def timed(label, step):
    """Run one pipeline step, printing a header and how long it took."""
    print(f"\n########## {label}", flush=True)
    start = time.time()
    step()
    print(f"########## {label} done in {time.time() - start:.0f}s", flush=True)


def first_phase():
    """Select the key frames, reconstruct both videos in 3D and extract DINOv3 features."""
    import features
    import frames
    import reconstruction_3d

    timed("1/9 key frames", frames.extract_all)
    timed("2/9 3D reconstruction (VGGT-Omega)", reconstruction_3d.reconstruct_all)
    timed("3/9 DINOv3 features", features.extract_all)


def second_phase():
    """Fuse, align, segment and write the verdict clouds; everything here uses the scene-scaled voxel size."""
    import attach
    import label_objects
    import match_changes
    import register
    import segment
    import tsdf

    timed("4/9 TSDF fusion", tsdf.fuse_all)
    timed("5/9 ICP alignment", register.export_registration)
    timed("6/9 attach features to points", attach.attach_all)
    timed("7/9 segmentation", segment.segment_all)
    timed("8/9 detect and match changes", match_changes.verdict_all)
    if os.environ.get("PIPELINE_LABELS", "1") != "0":
        timed("9/9 name the changed objects (VLM)", label_objects.label_all)


def main():
    """Run the pipeline on the given pair and report where the final clouds are stored."""
    args = parse_args()
    before, after = args.before.resolve(), args.after.resolve()
    if args.no_labels:
        os.environ["PIPELINE_LABELS"] = "0"
    out_dir = ROOT / "output" / (args.name or run_name(before, after))
    os.environ.update(PIPELINE_BEFORE=str(before), PIPELINE_AFTER=str(after), PIPELINE_OUTPUT=str(out_dir))

    if args.phase == 1:
        print(f"Run: {out_dir}")
        describe_videos(before, after)
        timed("0/9 clean videos", lambda: clean_inputs(before, after, out_dir))
        before, after = cleaned_path(out_dir, "before"), cleaned_path(out_dir, "after")
        os.environ.update(PIPELINE_BEFORE=str(before), PIPELINE_AFTER=str(after))
        first_phase()
        os.environ["PIPELINE_VOXEL"] = f"{scene_voxel(out_dir):.5f}"
        print(f"\nScene-scaled voxel size: {os.environ['PIPELINE_VOXEL']}")
        # the remaining stages read the voxel size when they are imported, so they run in a fresh process
        command = [sys.executable, str(Path(__file__).resolve()), str(before), str(after), "--phase", "2", "--name", out_dir.name]
        subprocess.run(command, check=True)
    else:
        second_phase()
        return

    print("\nResults:")
    for path in [out_dir / "diagnose" / "before_verdict_colour.ply", out_dir / "diagnose" / "after_verdict_colour.ply", out_dir / "diagnose" / "verdict_colour_best_frames.jpg", out_dir / "diagnose" / "verdicts.json"]:
        print(f"  {path}")
    print(f"Colours: {STATUS_LEGEND}")


if __name__ == "__main__":
    main()
