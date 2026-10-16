"""Copy each finished scene (cleaned videos plus verdict clouds) and the pipeline code into space/ for the demo."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "space"))

from clouds import read_ply, save_npz  # noqa: E402

OUTPUT, SPACE = ROOT / "output", ROOT / "space"
EXAMPLES = SPACE / "examples"
EXCLUDED = {"timing_test", "laundry_room_7_laundry_room_8"}  # a duplicate run, and a scene that is not a good example


def finished_scenes():
    """Return the scene names that have both verdict clouds and both cleaned videos."""
    names = [p.parent.parent.name for p in OUTPUT.glob("*/diagnose/before_verdict_colour.ply")]
    return sorted(n for n in names if n not in EXCLUDED and (OUTPUT / n / "input" / "before.mp4").exists() and (OUTPUT / n / "input" / "after.mp4").exists())


def build_scene(name):
    """Create space/examples/<name>/ with both videos, both converted clouds and the verdicts."""
    folder = EXAMPLES / name
    folder.mkdir(parents=True, exist_ok=True)
    for video in ("before", "after"):
        shutil.copy(OUTPUT / name / "input" / f"{video}.mp4", folder / f"{video}.mp4")
        save_npz(*read_ply(OUTPUT / name / "diagnose" / f"{video}_verdict_colour.ply"), folder / f"{video}.npz")
    shutil.copy(OUTPUT / name / "diagnose" / "verdicts.json", folder / "verdicts.json")  # the changed objects, their verdicts and VLM labels


def copy_pipeline():
    """Copy the pipeline modules the live tab runs into space/pipeline/."""
    shutil.rmtree(SPACE / "pipeline", ignore_errors=True)
    shutil.copytree(ROOT / "pipeline", SPACE / "pipeline", ignore=shutil.ignore_patterns("__pycache__", "diagnose.py", "clean_video.py"))


def main():
    """Rebuild the pipeline copy and every finished scene, replacing the old examples; rerun after adding or changing a scene."""
    copy_pipeline()
    shutil.rmtree(EXAMPLES, ignore_errors=True)
    for name in finished_scenes():
        build_scene(name)
        print("built", name)


if __name__ == "__main__":
    main()
