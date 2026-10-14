"""Comparison stage: decides, for every segmented object, whether it was added, removed, moved or unchanged between the two videos, judging an object only where the other video could actually see its location (depth-map occlusion test) and matching removed/added candidates by DINOv3 descriptors."""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from attach import project
from config import (
    ALIGNMENT_DIR,
    CHECK_FRAME,
    COMPARE_DIR,
    FEATURES_DIR,
    KEEP_FRACTIONS,
    MATCH_SIM,
    MATCH_VOTE,
    MIN_OBSERVED_FRACTION,
    MIN_VIEW_FRACTION,
    MOVE_DISTANCE,
    PRESENT_RADIUS,
    SEE_THROUGH_MARGIN,
    SEGMENTS_DIR,
)
from point_clouds import load_reconstruction
from segment import render_check
from tsdf import keep_mask

COLORS = {
    "background": (0.55, 0.55, 0.55),
    "unchanged": (0.95, 0.90, 0.50),
    "unobserved": (0.30, 0.30, 0.30),
    "added": (0.55, 0.85, 0.55),
    "removed": (0.95, 0.55, 0.55),
    "moved": (0.55, 0.65, 0.95),
}


def load_video(name):
    """Load a video's points, features, segment labels and reconstruction."""
    data = np.load(FEATURES_DIR / f"{name}.npz")
    return {
        "points": data["points"],
        "features": data["features"].astype(np.float32),
        "labels": np.load(SEGMENTS_DIR / f"{name}_labels.npy"),
        "recon": load_reconstruction(name),
    }


def point_status(points, other):
    """Classify points (in the other video's own coordinates) as 1 present, -1 absent (seen, empty) or 0 unobserved, using the other video's object points and depth maps."""
    other_points = other["points"][other["labels"] > 0]
    present = KDTree(other_points).query(points)[0] <= PRESENT_RADIUS

    recon = other["recon"]
    n_frames, height, width = recon["depth"].shape[:3]
    usable = keep_mask(recon["depth_conf"], KEEP_FRACTIONS[0])
    see_through = np.zeros(len(points), dtype=int)
    blocked = np.zeros(len(points), dtype=int)
    for i in range(n_frames):
        u, v, z, inside = project(points, recon["extrinsic"][i], recon["intrinsic"][i], height, width)
        ok = inside & usable[i][v, u]
        depth = recon["depth"][i][..., 0][v, u]
        see_through += ok & (depth > z + SEE_THROUGH_MARGIN)  # the camera saw past this spot, so nothing was there
        blocked += ok & (depth < z - SEE_THROUGH_MARGIN)  # something nearer hid this spot

    status = np.zeros(len(points), dtype=int)
    status[(see_through >= MIN_VIEW_FRACTION * n_frames) & (see_through > blocked)] = -1  # absent only if enough frames saw it empty
    status[present] = 1
    return status


def summarise_objects(video, status, centroids_frame):
    """Return one record per object: size, centroid, share of present/absent/unobserved points and a first verdict."""
    objects = []
    for label in np.unique(video["labels"][video["labels"] > 0]):
        members = video["labels"] == label
        s = status[members]
        present, absent = (s == 1).sum(), (s == -1).sum()
        observed = present + absent
        if observed < MIN_OBSERVED_FRACTION * len(s):
            verdict = "unobserved"
        elif present / observed >= 0.5:
            verdict = "unchanged"
        else:
            verdict = "candidate"
        objects.append({
            "label": int(label), "indices": np.where(members)[0], "n": int(members.sum()),
            "centroid": centroids_frame[members].mean(axis=0),
            "present": present / len(s), "absent": absent / len(s), "unobserved": (s == 0).mean(),
            "verdict": verdict,
        })
    return objects


def descriptor_match(features_a, features_b, rng, sample=1000):
    """Return the smaller of the two shares of points (A to B, B to A) that have a partner with cosine similarity of at least MATCH_SIM."""
    a = features_a[rng.choice(len(features_a), min(sample, len(features_a)), replace=False)]
    b = features_b[rng.choice(len(features_b), min(sample, len(features_b)), replace=False)]
    similarity = a @ b.T
    return min((similarity.max(axis=1) >= MATCH_SIM).mean(), (similarity.max(axis=0) >= MATCH_SIM).mean())


def match_candidates(before, after, before_objects, after_objects):
    """Pair removed candidates (before) with added candidates (after) by descriptor vote; return pairs sorted by score."""
    rng = np.random.default_rng(0)
    scores = []
    for i, b in enumerate(before_objects):
        for j, a in enumerate(after_objects):
            if b["verdict"] == "candidate" and a["verdict"] == "candidate":
                score = descriptor_match(before["features"][b["indices"]], after["features"][a["indices"]], rng)
                if score >= MATCH_VOTE:
                    scores.append((score, i, j))
    pairs, used_b, used_a = [], set(), set()
    for score, i, j in sorted(scores, reverse=True):
        if i not in used_b and j not in used_a:
            pairs.append((i, j, score))
            used_b.add(i)
            used_a.add(j)
    return pairs


def compare_videos():
    """Classify every object, write status clouds and a check image to output/compare/, and print an object table."""
    before, after = load_video("before"), load_video("after")
    T = np.load(ALIGNMENT_DIR / "after_to_before.npy")
    rotation, translation = T[:3, :3], T[:3, 3]
    after_in_before = after["points"] @ rotation.T + translation  # after points in before's coordinates
    before_in_after = (before["points"] - translation) @ rotation  # before points in after's own coordinates

    before_status = np.zeros(len(before["points"]), dtype=int)
    after_status = np.zeros(len(after["points"]), dtype=int)
    b_idx, a_idx = np.where(before["labels"] > 0)[0], np.where(after["labels"] > 0)[0]
    before_status[b_idx] = point_status(before_in_after[b_idx], after)
    after_status[a_idx] = point_status(after_in_before[a_idx], before)

    before_objects = summarise_objects(before, before_status, before["points"])
    after_objects = summarise_objects(after, after_status, after_in_before)
    pairs = match_candidates(before, after, before_objects, after_objects)

    verdicts = {"before": {}, "after": {}}
    for obj in before_objects:
        verdicts["before"][obj["label"]] = "removed" if obj["verdict"] == "candidate" else obj["verdict"]
    for obj in after_objects:
        verdicts["after"][obj["label"]] = "added" if obj["verdict"] == "candidate" else obj["verdict"]

    colors = {name: np.tile(COLORS["background"], (len(v["points"]), 1)) for name, v in (("before", before), ("after", after))}
    for name, video, objects in (("before", before, before_objects), ("after", after, after_objects)):
        for obj in objects:
            colors[name][obj["indices"]] = COLORS[verdicts[name][obj["label"]]]

    print("\nmoved / same-place pairs (descriptor vote, centre displacement):")
    for i, j, score in pairs:
        b, a = before_objects[i], after_objects[j]
        distance = float(np.linalg.norm(a["centroid"] - b["centroid"]))
        verdict = "moved" if distance >= MOVE_DISTANCE else "unchanged"
        verdicts["before"][b["label"]] = verdicts["after"][a["label"]] = verdict
        colors["before"][b["indices"]] = colors["after"][a["indices"]] = COLORS[verdict]
        print(f"  before obj {b['label']:2d} <-> after obj {a['label']:2d}: match {score:.2f}, displacement {distance:.3f} -> {verdict}")

    print("\nobjects (share of points present / absent / unobserved in the other video):")
    for name, objects in (("before", before_objects), ("after", after_objects)):
        for obj in objects:
            c = obj["centroid"]
            print(f"  {name:6s} obj {obj['label']:2d}: {obj['n']:5d} pts at ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})  "
                  f"{obj['present']:4.0%} / {obj['absent']:4.0%} / {obj['unobserved']:4.0%}  -> {verdicts[name][obj['label']]}")

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    for name, video in (("before", before), ("after", after)):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(video["points"])
        cloud.colors = o3d.utility.Vector3dVector(colors[name])
        o3d.io.write_point_cloud(str(COMPARE_DIR / f"{name}_status.ply"), cloud)
    render_check({name: COMPARE_DIR / f"{name}_status.ply" for name in ("before", "after")}, CHECK_FRAME,
                 COMPARE_DIR / f"check_frame{CHECK_FRAME}.jpg")
