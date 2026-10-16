"""Frame extraction stage: turns each input video into a small, evenly spread set of sharp JPEG frames with an index describing which source frames were chosen."""

import json

import cv2
import numpy as np

from config import BIN_SECONDS, DATA, FRAMES_DIR, MAX_BINS, MIN_BINS, MIN_GAP_FRAMES, VIDEO_FILES, VIDEOS


def sharpness(frame):
    """Return the variance of the Laplacian of a frame (higher means sharper)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def score_frames(video_path):
    """Return the sharpness score of every frame in a video, plus its fps."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    scores = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scores.append(sharpness(frame))
    cap.release()
    return np.array(scores), fps


def pick_indices(scores, n_bins, min_gap):
    """Pick one frame index per time bin: the sharpest one at least min_gap frames after the previous pick."""
    edges = np.linspace(0, len(scores), n_bins + 1).astype(int)
    min_gap = min(min_gap, len(scores) // n_bins // 2)  # keep the gap well below the bin length
    picks = []
    prev = -min_gap
    for b in range(n_bins):
        start = max(edges[b], prev + min_gap)
        idx = int(start + np.argmax(scores[start:edges[b + 1]]))
        picks.append(idx)
        prev = idx
    return picks


def read_frames(video_path, indices):
    """Return {frame index: image} for the requested frame indices."""
    wanted = set(indices)
    cap = cv2.VideoCapture(str(video_path))
    found = {}
    for idx in range(max(wanted) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            found[idx] = frame
    cap.release()
    return found


def choose_n_bins(n_frames, fps):
    """Return the number of time bins for a video: one per BIN_SECONDS of footage, kept between MIN_BINS and MAX_BINS."""
    return int(np.clip(round(n_frames / fps / BIN_SECONDS), MIN_BINS, MAX_BINS))


def extract_frames(name, n_bins=None, min_gap=MIN_GAP_FRAMES):
    """Select and save the chosen frames of one video as JPEGs, and write index.json."""
    out_dir = FRAMES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = DATA / VIDEO_FILES[name]

    scores, fps = score_frames(video_path)
    n_bins = n_bins or choose_n_bins(len(scores), fps)
    picks = pick_indices(scores, n_bins, min_gap)
    images = read_frames(video_path, picks)

    index = []
    for b, idx in enumerate(picks):
        cv2.imwrite(str(out_dir / f"{b:03d}.jpg"), images[idx], [cv2.IMWRITE_JPEG_QUALITY, 95])
        index.append({"bin": b, "frame": idx, "time_s": idx / fps, "sharpness": float(scores[idx])})

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return index


def extract_all():
    """Run frame extraction for every video in config.VIDEOS."""
    return {name: extract_frames(name) for name in VIDEOS}
