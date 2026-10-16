"""Gradio demo: browse example scenes or run the pipeline on two uploaded videos."""

import json
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

from clouds import load_npz
from live import MAX_SECONDS, run_live

EXAMPLES = Path(__file__).parent / "examples"
CAMERA = dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0.3, y=-0.2, z=-2.0))  # VGGT frame: y points down, so view from behind the first camera
LEGEND = "Original colours; changed objects are tinted and named by a small vision-language model: blue = moved · green = added · red = removed"
LIVE_NOTE = (
    f"Upload a before and an after video of the same scene (at most {MAX_SECONDS} seconds each, nothing private). "
    "It takes a few minutes. This runs on free ZeroGPU: open this page on huggingface.co and sign in to get 5 GPU minutes a day instead of 2."
)


FIRST = ("dining_table", "scene_diff_living_room", "desk")  # shown first, in this order; the rest follow alphabetically


def scene_names():
    """Return the scene folder names under examples/, the FIRST ones leading."""
    names = sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir())
    return [n for n in FIRST if n in names] + [n for n in names if n not in FIRST]


def side_records(records, side):
    """Return the changed objects visible in one video: removed and moved ones in before, added and moved ones in after."""
    return [r for r in records[side] if r["verdict"] != ("added" if side == "before" else "removed")]


def changes_table(records):
    """Return the plain-English change list as Markdown bullets, one per changed object."""
    sentences = records.get("summary", [])
    return "\n".join(f"- {sentence}" for sentence in sentences) if sentences else "No changes found."


def cloud_figure(xyz, rgb, records=()):
    """Build an interactive 3D scatter of a colour point cloud, with each changed object's label at its centre."""
    colours = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    fig = go.Figure(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", marker=dict(size=1.5, color=colours), hoverinfo="skip", showlegend=False))
    if records:
        centres = [r["centroid"] for r in records]
        fig.add_trace(go.Scatter3d(x=[c[0] for c in centres], y=[c[1] for c in centres], z=[c[2] for c in centres], mode="text", showlegend=False, hoverinfo="skip",
                                   text=[f"{r.get('label', '?')} ({r['verdict']})" for r in records], textfont=dict(size=14, color="black")))
    fig.update_layout(scene=dict(aspectmode="data", camera=CAMERA), margin=dict(l=0, r=0, t=0, b=0), height=450)
    return fig


def show_scene(name):
    """Return the before and after videos, clouds and the list of changes of one example scene."""
    folder = EXAMPLES / name
    records = json.loads((folder / "verdicts.json").read_text())
    return (str(folder / "before.mp4"), str(folder / "after.mp4"), cloud_figure(*load_npz(folder / "before.npz"), side_records(records, "before")),
            cloud_figure(*load_npz(folder / "after.npz"), side_records(records, "after")), changes_table(records))


def live_figures(before, after):
    """Run the pipeline on the uploaded videos and return the two 3D figures and the list of changes."""
    before_cloud, after_cloud, records = run_live(before, after)
    return cloud_figure(*before_cloud, side_records(records, "before")), cloud_figure(*after_cloud, side_records(records, "after")), changes_table(records)


def build_examples_tab():
    """Lay out the scene picker, videos and 3D views of the examples."""
    names = scene_names()
    picker = gr.Dropdown(names, value=names[0], label="Scene")
    with gr.Row():
        before_video, after_video = gr.Video(label="Before"), gr.Video(label="After")
    with gr.Row():
        before_cloud, after_cloud = gr.Plot(label="Before (3D)"), gr.Plot(label="After (3D)")
    table = gr.Markdown()
    outputs = [before_video, after_video, before_cloud, after_cloud, table]
    picker.change(show_scene, picker, outputs)
    return picker, outputs


def build_live_tab():
    """Lay out the two upload boxes, the run button and the two result views."""
    gr.Markdown(LIVE_NOTE)
    with gr.Row():
        before, after = gr.Video(label="Before", sources=["upload"]), gr.Video(label="After", sources=["upload"])
    button = gr.Button("Run", variant="primary")
    with gr.Row():
        before_cloud, after_cloud = gr.Plot(label="Before (3D)"), gr.Plot(label="After (3D)")
    table = gr.Markdown()
    button.click(live_figures, [before, after], [before_cloud, after_cloud, table])


def build_demo():
    """Lay out the title, legend and the two tabs."""
    with gr.Blocks(title="Humanoid Spatial") as demo:
        gr.Markdown(f"# Scene Change Detection\n{LEGEND}")
        with gr.Tab("Examples"):
            picker, outputs = build_examples_tab()
        with gr.Tab("Try Your Own"):
            build_live_tab()
        demo.load(show_scene, picker, outputs)
    return demo


if __name__ == "__main__":
    build_demo().queue(default_concurrency_limit=1).launch()
