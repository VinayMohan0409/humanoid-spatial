"""Live tab: run the full pipeline on two uploaded videos and return their verdict point clouds."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import gradio as gr
import spaces

from clouds import read_ply

APP = Path(__file__).parent
PIPELINE = APP / "pipeline"
OUT = APP / "output" / "live"
BEFORE, AFTER = OUT / "before.mp4", OUT / "after.mp4"
MAX_SECONDS = 22
GPU_SECONDS = 120
LABEL_SECONDS = 60

os.environ.update(PIPELINE_BEFORE=str(BEFORE), PIPELINE_AFTER=str(AFTER), PIPELINE_OUTPUT=str(OUT))
sys.path.insert(0, str(PIPELINE))


def download_weights():
    """Fetch the gated VGGT-Omega and DINOv3 weights and the (open) Qwen vision-language model at startup so the download does not use GPU time."""
    from config import CHECKPOINT_FILE, CHECKPOINT_REPO, DINO_MODEL, VLM_MODEL
    from huggingface_hub import hf_hub_download, snapshot_download

    hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE)
    snapshot_download(DINO_MODEL)
    snapshot_download(VLM_MODEL)


def check_video(path):
    """Raise a user-facing error if a video cannot be read or is longer than MAX_SECONDS."""
    cap = cv2.VideoCapture(path)
    frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or frames / fps > MAX_SECONDS:
        raise gr.Error(f"Each video must be readable and at most {MAX_SECONDS} seconds long.")


def stage_inputs(before, after):
    """Empty the work folder and copy the two uploaded videos into it."""
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    shutil.copy(before, BEFORE)
    shutil.copy(after, AFTER)


@spaces.GPU(duration=GPU_SECONDS)
def gpu_stages():
    """Run key-frame selection, VGGT-Omega reconstruction and DINOv3 features on the GPU."""
    import main

    main.first_phase()


def cpu_stages():
    """Run the remaining stages except the naming in a fresh process, which picks up the scene-scaled voxel size."""
    import main

    os.environ["PIPELINE_VOXEL"] = f"{main.scene_voxel(OUT):.5f}"
    subprocess.run([sys.executable, str(PIPELINE / "main.py"), str(BEFORE), str(AFTER), "--name", "live", "--phase", "2", "--no-labels"], check=True)


@spaces.GPU(duration=LABEL_SECONDS)
def gpu_labels():
    """Name the changed objects with the vision-language model on the GPU."""
    import label_objects

    label_objects.label_all()


def run_live(before, after, progress=gr.Progress()):
    """Process two uploaded videos and return the before and after verdict clouds and the verdicts with their labels."""
    if before is None or after is None:
        raise gr.Error("Upload both a before and an after video.")
    check_video(before)
    check_video(after)
    stage_inputs(before, after)
    progress(0.1, desc="Reconstructing in 3D on the GPU")
    gpu_stages()
    progress(0.6, desc="Comparing the scenes")
    cpu_stages()
    progress(0.85, desc="Naming the changed objects")
    gpu_labels()
    clouds = [read_ply(OUT / "diagnose" / f"{name}_verdict_colour.ply") for name in ("before", "after")]
    records = json.loads((OUT / "diagnose" / "verdicts.json").read_text())
    shutil.rmtree(OUT, ignore_errors=True)
    return [*clouds, records]


try:
    download_weights()
except Exception as error:
    print(f"Live tab unavailable, weights not downloaded: {error}")
