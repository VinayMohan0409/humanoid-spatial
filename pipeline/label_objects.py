"""Naming stage (also writes the plain-English change list and the labelled result image): crops each changed unit out of the key frames where it is most visible, asks a small vision-language model what the object is (one word, or "unknown"), and writes the label into verdicts.json with the crops that were shown (diagnose/labels/).
A moved object is seen in both videos, so its views are pooled and both sides get the same label. Decoding is greedy, so the answer does not change between runs."""

import json
import re
import sys
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image

from attach import project
from config import FEATURES_DIR, OUTPUT, VIDEOS, VISIBILITY_TOL, VLM_MARGIN, VLM_MIN_VISIBLE, VLM_MODEL, VLM_PROMPT, VLM_VIEWS
from point_clouds import load_reconstruction
from reconstruction_3d import get_device

NO_ANSWER = {"unknown", "unsure", "unclear", "none", "nothing", "object", "thing", "item", "unidentified"}
FILLER = {"a", "an", "the", "this", "that", "it", "is", "its"}
TAG_FONT, TAG_SCALE = cv2.FONT_HERSHEY_SIMPLEX, 0.5
MIN_SIDE = 224  # crops smaller than this are enlarged, so a small object still fills the model's input


def load_vlm(device):
    """Load the vision-language model and its processor (float32 on CPU and Apple GPUs, bfloat16 on CUDA)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(VLM_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(VLM_MODEL, dtype=dtype).to(device).eval()
    return processor, model


def unit_views(recon, points):
    """Return up to VLM_VIEWS (frame, visible points, RGB crop) for a unit, from the frames that show most of its points, most visible first. A point counts as visible if the depth map agrees it is the nearest surface."""
    n_frames, height, width = recon["depth"].shape[:3]
    seen, boxes = [], []
    for i in range(n_frames):
        u, v, z, inside = project(points, recon["extrinsic"][i], recon["intrinsic"][i], height, width)
        depth = recon["depth"][i][..., 0][v, u]
        visible = inside & (np.abs(z - depth) / np.maximum(depth, 1e-6) < VISIBILITY_TOL)
        seen.append(int(visible.sum()))
        boxes.append(None if visible.sum() < VLM_MIN_VISIBLE else (np.percentile(u[visible], (2, 98)), np.percentile(v[visible], (2, 98))))
    views = []
    for i in np.argsort(-np.asarray(seen), kind="stable")[:VLM_VIEWS]:
        if boxes[i] is None:
            continue
        (x0, x1), (y0, y1) = boxes[i]
        pad = VLM_MARGIN * max(x1 - x0, y1 - y0) + 6
        x0, x1, y0, y1 = int(max(x0 - pad, 0)), int(min(x1 + pad, width)), int(max(y0 - pad, 0)), int(min(y1 + pad, height))
        crop = recon["images"][i].transpose(1, 2, 0)[y0:y1, x0:x1]
        scale = max(1.0, MIN_SIDE / max(crop.shape[:2]))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        views.append((int(i), seen[i], np.ascontiguousarray(crop)))
    return views


def clean_answer(text):
    """Return the first meaningful lowercase word of the model's reply, or "unknown"."""
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in FILLER]
    return words[0] if words and words[0] not in NO_ANSWER else "unknown"


def ask(processor, model, device, crop):
    """Show the model one crop and return its one-word answer."""
    messages = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(crop)}, {"type": "text", "text": VLM_PROMPT}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    return clean_answer(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])


def vote(answers):
    """Return the most common informative answer among (answer, visible points) pairs; ties go to the answer from the most visible view; "unknown" if there is none."""
    informative = [(a, n) for a, n in answers if a != "unknown"]
    if not informative:
        return "unknown"
    counts = Counter(a for a, _ in informative)
    best = max(counts.values())
    return max((x for x in informative if counts[x[0]] == best), key=lambda x: x[1])[0]


def sentence(record):
    """Return a plain-English sentence for one changed object, e.g. "A mouse was moved." """
    label = record["label"]
    noun = "An unidentified object" if label == "unknown" else f"{'An' if label[0] in 'aeiou' else 'A'} {label}"
    if record["verdict"] == "moved":
        return f"{noun if label == 'unknown' else 'The ' + label} was moved."
    return f"{noun} was {record['verdict']}."


def describe(records):
    """Return one sentence per changed object (a moved object once, from the before side)."""
    return [sentence(r) for r in records["before"]] + [sentence(r) for r in records["after"] if r["verdict"] == "added"]


def draw_tag(image, box, text):
    """Draw a box with a white tag and black text above it (readable on any surface) on a BGR image, in place."""
    x0, y0, x1, y1 = (int(v) for v in box)
    cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), 1, cv2.LINE_AA)
    (w, h), base = cv2.getTextSize(text, TAG_FONT, TAG_SCALE, 1)
    top = y0 - h - base - 4 if y0 - h - base - 4 >= 0 else y1 + 2
    left = int(np.clip(x0, 0, image.shape[1] - w - 6))
    cv2.rectangle(image, (left, top), (left + w + 6, top + h + base + 4), (255, 255, 255), -1)
    cv2.rectangle(image, (left, top), (left + w + 6, top + h + base + 4), (0, 0, 0), 1)
    cv2.putText(image, text, (left + 3, top + h + 1), TAG_FONT, TAG_SCALE, (0, 0, 0), 1, cv2.LINE_AA)


def annotate_all():
    """From the labels in output/diagnose/verdicts.json, write the sentences (verdicts.json "summary", changes.txt) and diagnose/verdict_labelled.jpg: the best-frame image with every changed object boxed and tagged "label - verdict" on both the camera frame and the cloud. Needs no model."""
    from match_changes import best_frame

    diagnose = OUTPUT / "diagnose"
    records = json.loads((diagnose / "verdicts.json").read_text())
    plain = diagnose / "verdict_colour_best_frames_plain.jpg"
    if not plain.exists():  # keep the unlabelled picture, so this can be rerun
        (diagnose / "verdict_colour_best_frames.jpg").rename(plain)
    image = cv2.imread(str(plain))
    rows = []
    for row, name in enumerate(VIDEOS):
        recon = load_reconstruction(name)
        points = np.load(FEATURES_DIR / f"{name}.npz")["points"]
        unit_map = np.load(diagnose / f"{name}_units.npy")
        height, width = recon["depth"].shape[1:3]
        original = image[row * height:(row + 1) * height, :width].copy()
        verdict = image[row * height:(row + 1) * height, width:2 * width].copy()
        for panel, caption in ((original, "original"), (verdict, "verdict")):  # replace the pipeline's own captions, which are hard to read on light pictures
            cv2.rectangle(panel, (0, 0), (340, 34), (0, 0, 0), -1)
            cv2.putText(panel, f"{name}: {caption}", (6, 22), TAG_FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        labelled = verdict.copy()
        cv2.rectangle(labelled, (0, 0), (340, 34), (0, 0, 0), -1)
        cv2.putText(labelled, f"{name}: verdict + labels", (6, 22), TAG_FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        frame = best_frame({"points": points, "recon": recon}, unit_map > 0)
        for record in records[name]:
            u, v, z, inside = project(points[unit_map == record["unit"]], recon["extrinsic"][frame], recon["intrinsic"][frame], height, width)
            depth = recon["depth"][frame][..., 0][v, u]
            seen = inside & (np.abs(z - depth) / np.maximum(depth, 1e-6) < VISIBILITY_TOL)
            if seen.sum() < 5:
                continue  # this object is not visible in this video's best frame
            (x0, x1), (y0, y1) = np.percentile(u[seen], (2, 98)), np.percentile(v[seen], (2, 98))
            draw_tag(labelled, (x0, y0, x1, y1), f"{record['label']} - {record['verdict']}")
        rows.append(np.hstack([original, verdict, labelled]))
    cv2.imwrite(str(diagnose / "verdict_labelled.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    records["summary"] = describe(records)
    (diagnose / "verdicts.json").write_text(json.dumps(records, indent=2))
    (diagnose / "changes.txt").write_text("\n".join(records["summary"]) + "\n")
    print("\nchanges:\n  " + "\n  ".join(records["summary"]))


def label_all():
    """Name every changed unit in output/diagnose/verdicts.json, add a "label" to each record and save the crops that were shown to output/diagnose/labels/."""
    diagnose = OUTPUT / "diagnose"
    records = json.loads((diagnose / "verdicts.json").read_text())
    (diagnose / "labels").mkdir(exist_ok=True)
    device = get_device()
    processor, model = load_vlm(device)

    answers = {}  # (video, unit id) -> [(answer, visible points)]
    for name in VIDEOS:
        recon = load_reconstruction(name)
        points = np.load(FEATURES_DIR / f"{name}.npz")["points"]
        unit_map = np.load(diagnose / f"{name}_units.npy")
        for record in records[name]:
            found = []
            for j, (frame, visible, crop) in enumerate(unit_views(recon, points[unit_map == record["unit"]])):
                found.append((ask(processor, model, device, crop), visible))
                cv2.imwrite(str(diagnose / "labels" / f"{name}_unit{record['unit']}_view{j}_{found[-1][0]}.jpg"), crop[:, :, ::-1])
            answers[(name, record["unit"])] = found

    for name in VIDEOS:
        other = "after" if name == "before" else "before"
        for record in records[name]:
            pooled = list(answers[(name, record["unit"])])
            if record["partner_unit"] is not None:  # the same object seen in both videos
                pooled += answers[(other, record["partner_unit"])]
            record["label"] = vote(pooled)
            record["views"] = [a for a, _ in answers[(name, record["unit"])]]
    (diagnose / "verdicts.json").write_text(json.dumps(records, indent=2))

    print(f"\nlabels from {VLM_MODEL}:")
    for name in VIDEOS:
        for record in records[name]:
            print(f"  {name:6s} unit {record['unit']:4d} {record['verdict']:8s} {record['points']:6d} pts -> {record['label']:12s} (views: {', '.join(record['views']) or 'none visible'})")
    annotate_all()


if __name__ == "__main__":
    annotate_all() if "--annotate-only" in sys.argv else label_all()
