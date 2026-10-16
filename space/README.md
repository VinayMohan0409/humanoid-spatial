---
title: Humanoid Spatial
emoji: 🧊
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12.12
app_file: app.py
pinned: false
---

Detects added, removed and moved objects between a before and an after video of a scene, names them with a small vision-language model, and shows the 3D reconstructions with the changed objects tinted (blue = moved, green = added, red = removed).

The two videos are reconstructed together in 3D, every point is flagged as changed or not, the points are grouped into objects (large planes are peeled off, the rest is grown by proximity and DINOv3 feature similarity), and objects are matched across the two videos by how they look.
