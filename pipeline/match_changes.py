"""Matching stage: turns segmented objects and change regions into per-video units, describes each, assigns before units to after units by identity, and labels every changed unit moved, removed or added."""

import json

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree

from appearance import lab_ab
from attach import project
from change_regions import change_regions
from compare import COLORS, aligned_points, load_video
from config import ABSORB_MAX_RATIO, CHECK_FRAME, CHROMA_BINS, DINO_MIN, IDENTITY_MIN, MOVE_DISTANCE, OBJECT_MIN_POINTS, OUTPUT, RING_DISTINCT_SHARE, RING_MAX, RING_POINT_SIM, RING_RADIUS, UNIT_FLAG_SHARE, VISIBILITY_TOL, Z_WEAK
from point_clouds import make_cloud
from segment import render_check

OVERLAY_COLORS = {"moved": (0.1, 0.3, 1.0), "added": (0.1, 0.85, 0.2), "removed": (1.0, 0.1, 0.1)}  # vivid, to stand out against the true colours
RING_MODE = "mean"  # "mean": a unit is an empty footprint if its mean feature matches its surroundings'; "points": if few of its own points differ from them (finds more small objects, but also false ones; tested on desk, dining_table and living_room)
OVERLAY_ALPHA = 0.5  # share of the verdict colour blended over a changed unit's true colour


def unit_ids(video, evidence):
    """Return a unit id per point: segmented objects keep their label; a change region becomes its own unit unless it lies mostly inside one segment of comparable size (at most ABSORB_MAX_RATIO times the region), so a small changed object is not swallowed by the bed or desk it sits on."""
    segments, regions = video["labels"], evidence["labels"]
    units = segments.copy()
    for k in range(1, regions.max() + 1):
        members = regions == k
        counts = np.bincount(segments[members])
        counts[0] = 0
        top = counts.argmax()
        if counts[top] >= 0.5 * members.sum():
            if (segments == top).sum() <= ABSORB_MAX_RATIO * members.sum():
                continue
            members &= (segments == 0) | (segments == top)
        else:
            members &= segments == 0
        units[members] = units.max() + 1
    return units


def describe(video, indices, points):
    """Return an object descriptor: unit mean DINO feature, Lab a,b histogram and the sorted spreads of its points along their principal axes."""
    feature = video["features"][indices].mean(axis=0)
    ab = lab_ab(video["colors"][indices])
    hist = np.histogram2d(ab[:, 0], ab[:, 1], bins=CHROMA_BINS, range=[[-60, 60], [-60, 60]])[0]
    spread = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(points[indices].T)), 1e-12))
    return {"feature": feature / np.linalg.norm(feature), "hist": hist / hist.sum(), "spread": spread}


def ring_features(video, tree, indices):
    """Return the unit-length mean feature of the points around a unit (within RING_RADIUS, excluding itself), or None if there are none."""
    near = np.unique(np.concatenate(tree.query_ball_point(video["points"][indices][::3], RING_RADIUS)))
    outside = np.setdiff1d(near, indices)
    if len(outside) == 0:
        return None  # nothing around it, so it cannot be an empty footprint
    ring = video["features"][outside].mean(axis=0)
    return ring / np.linalg.norm(ring)


def surrounding_similarity(video, tree, indices, feature):
    """Return the cosine between a unit's mean feature and that of the points around it; near 1 means it is just more of its surroundings, like the empty table where an object was."""
    ring = ring_features(video, tree, indices)
    return 0.0 if ring is None else float(ring @ feature)


def distinct_share(video, tree, indices):
    """Return the share of a unit's points whose own feature differs from the surroundings' mean (cosine below RING_POINT_SIM). An empty footprint is surface like its surroundings, so almost no point differs; an object has many that do, even when its mean is diluted by patch blur or by neighbouring fragments of itself."""
    ring = ring_features(video, tree, indices)
    if ring is None:
        return 1.0
    f = video["features"][indices]
    return float(((f @ ring) / np.linalg.norm(f, axis=1) < RING_POINT_SIM).mean())


def build_units(video, evidence, common_points):
    """Return the changed units of a video (segments and flat change regions that differ from their surroundings) with their points, evidence and descriptor; common_points is the video's cloud in before's coordinates."""
    ids, tree = unit_ids(video, evidence), KDTree(video["points"])
    units = []
    for u in np.unique(ids[ids > 0]):
        indices = np.where(ids == u)[0]
        if len(indices) < OBJECT_MIN_POINTS:
            continue
        matched = evidence["matched"][indices]
        flagged = evidence["changed"][indices].mean()
        z = np.median(evidence["z_feature"][indices][matched]) if matched.mean() >= 0.5 else 0.0  # appearance is only comparable where the other video has surface
        if flagged < UNIT_FLAG_SHARE and z < Z_WEAK:
            continue
        descriptor = describe(video, indices, video["points"])
        ring = surrounding_similarity(video, tree, indices, descriptor["feature"])
        distinct = distinct_share(video, tree, indices)
        if (distinct >= RING_DISTINCT_SHARE) if RING_MODE == "points" else (ring <= RING_MAX):
            units.append({"id": int(u), "indices": indices, "n": len(indices), "centroid": common_points[indices].mean(axis=0),
                          "flagged": flagged, "geometric": evidence["geometric"][indices].mean(), "z": z, "ring": ring, "distinct": distinct, **descriptor})
    return units


def identity_score(a, b):
    """Return the mean of DINO cosine, Lab histogram intersection and principal-spread ratio, each in [0, 1], or 0 if the DINO cosine is below DINO_MIN."""
    cosine = a["feature"] @ b["feature"]
    ratio = np.minimum(a["spread"], b["spread"]) / np.maximum(a["spread"], b["spread"])
    return float(np.mean([cosine, np.minimum(a["hist"], b["hist"]).sum(), ratio.mean()])) if cosine >= DINO_MIN else 0.0


def match_units(before_units, after_units):
    """Return the score matrix (zero for pairs closer than MOVE_DISTANCE, which did not move) and the (before, after) index pairs of a one-to-one assignment over the pairs scoring at least IDENTITY_MIN."""
    score = np.zeros((len(before_units), len(after_units)))
    for i, b in enumerate(before_units):
        for j, a in enumerate(after_units):
            if np.linalg.norm(a["centroid"] - b["centroid"]) >= MOVE_DISTANCE:
                score[i, j] = identity_score(b, a)
    rows, cols = linear_sum_assignment(-np.where(score >= IDENTITY_MIN, score, 0.0))
    return score, [(i, j) for i, j in zip(rows, cols) if score[i, j] >= IDENTITY_MIN]


def best_frame(video, changed):
    """Return the key frame in which most of the changed points are visible (CHECK_FRAME if none are changed)."""
    if not changed.any():
        return CHECK_FRAME
    recon = video["recon"]
    height, width = recon["depth"].shape[1:3]
    points, visible = video["points"][changed], []
    for i in range(len(recon["depth"])):
        u, v, z, inside = project(points, recon["extrinsic"][i], recon["intrinsic"][i], height, width)
        depth = recon["depth"][i][..., 0][v, u]
        visible.append((inside & (np.abs(z - depth) / depth < VISIBILITY_TOL)).sum())
    return int(np.argmax(visible))


def render_best_frames(files, frames, out_path):
    """Save an image of each video at its own best frame: camera frame next to the coloured cloud, before on top, after below."""
    rows = []
    for k, name in enumerate(("before", "after")):
        render_check(files, frames[name], out_path)
        image = cv2.imread(str(out_path))
        rows.append(image[: image.shape[0] // 2] if k == 0 else image[image.shape[0] // 2 :])
    cv2.imwrite(str(out_path), np.vstack(rows))


def export_verdicts(before, after, out_dir):
    """Label every changed unit moved (matched pair), removed (unmatched before) or added (unmatched after), write verdict clouds (grey, and true colour with the changed units in vivid blue/green/red) with check images (the overlay shown at each video's best frame), and print the decisions."""
    after_in_before, before_in_after = aligned_points(before, after)
    evidence = {"before": change_regions(before, before_in_after, after, after_in_before),
                "after": change_regions(after, after_in_before, before, before_in_after)}
    units = {"before": build_units(before, evidence["before"], before["points"]),
             "after": build_units(after, evidence["after"], after_in_before)}
    score, pairs = match_units(units["before"], units["after"])
    verdict = {"before": ["removed"] * len(units["before"]), "after": ["added"] * len(units["after"])}
    partner = {"before": {i: j for i, j in pairs}, "after": {j: i for i, j in pairs}}
    for i, j in pairs:
        verdict["before"][i] = verdict["after"][j] = "moved"

    files, overlay_files, frames = {}, {}, {}
    records = {"before": [], "after": []}
    for name, video in (("before", before), ("after", after)):
        colors = np.tile(COLORS["unchanged"], (len(video["points"]), 1))
        overlay, changed = video["colors"].copy(), np.zeros(len(video["points"]), bool)
        unit_map = np.zeros(len(video["points"]), dtype=int)  # unit id per point (0 = not in a changed unit), for the naming stage
        other = "after" if name == "before" else "before"
        print(f"\n{name}: unit  points  flagged  geometric  feature z  ring  verdict  | best partner score (accepted if >= {IDENTITY_MIN})")
        for k, u in enumerate(units[name]):
            colors[u["indices"]] = COLORS[verdict[name][k]]
            overlay[u["indices"]] = (1 - OVERLAY_ALPHA) * overlay[u["indices"]] + OVERLAY_ALPHA * np.array(OVERLAY_COLORS[verdict[name][k]])
            changed[u["indices"]] = True
            unit_map[u["indices"]] = u["id"]
            row = score[k] if name == "before" else score[:, k]
            best = int(row.argmax()) if row.size else -1
            print(f"  {u['id']:4d} {u['n']:6d}  {u['flagged']:6.0%}  {u['geometric']:8.0%}  {u['z']:8.1f}  {u['ring']:.2f}  {verdict[name][k]:8s} | "
                  f"{other} unit {units[other][best]['id'] if best >= 0 else '-'}: {row[best] if best >= 0 else 0:.2f}"
                  f"{'  <- matched' if k in partner[name] else ''}")
            records[name].append({"unit": u["id"], "verdict": verdict[name][k], "points": int(u["n"]), "centroid": [round(float(x), 4) for x in u["centroid"]], "flagged": round(float(u["flagged"]), 3),
                                  "geometric": round(float(u["geometric"]), 3), "feature_z": round(float(u["z"]), 2), "partner_unit": units[other][partner[name][k]]["id"] if k in partner[name] else None})
        np.save(out_dir / f"{name}_units.npy", unit_map)
        files[name] = out_dir / f"{name}_verdict.ply"
        o3d.io.write_point_cloud(str(files[name]), make_cloud(video["points"], colors))
        frames[name] = best_frame(video, changed)
        overlay_files[name] = out_dir / f"{name}_verdict_colour.ply"
        o3d.io.write_point_cloud(str(overlay_files[name]), make_cloud(video["points"], overlay))
    render_check(files, CHECK_FRAME, out_dir / f"verdict_frame{CHECK_FRAME}.jpg")
    render_best_frames(overlay_files, frames, out_dir / "verdict_colour_best_frames.jpg")
    (out_dir / "verdicts.json").write_text(json.dumps(records, indent=2))


def verdict_all():
    """Write the verdict clouds of the scene selected by PIPELINE_OUTPUT into its diagnose folder."""
    out_dir = OUTPUT / "diagnose"
    out_dir.mkdir(exist_ok=True)
    export_verdicts(load_video("before"), load_video("after"), out_dir)
