# External models, data and libraries

Nothing below is redistributed in this repository except where stated. Model weights are downloaded from Hugging Face on first run.

## Models (gated: accept the terms on Hugging Face first)

| Component | Source | Licence | Use here |
|---|---|---|---|
| VGGT-Omega (code) | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega), pinned to `48b23c8` | FAIR Noncommercial Research License v1 | Multi-view 3D reconstruction (depth, camera poses) |
| VGGT-Omega (weights) | [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega), `vggt_omega_1b_512.pt` | FAIR Noncommercial Research License v1 | Same |
| DINOv3 ViT-B/16 | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | DINOv3 License (Meta) | Per-point semantic features |

- VGGT-Omega is **non-commercial, research use only**. Anyone replicating this work for commercial use needs their own licence from Meta.
- To replicate: request access to both repositories on Hugging Face, accept each licence, then `hf auth login`. The pipeline downloads the weights on first run.

## Data

| Data | Source | Licence | Use here |
|---|---|---|---|
| SceneDiff benchmark (six `scene_diff_*` example scenes) | [yuqun/SceneDiff](https://huggingface.co/datasets/yuqun/SceneDiff), Wu et al., ECCV 2026, [arXiv:2512.16908](https://arxiv.org/abs/2512.16908) | MIT (dataset card tag; code repository LICENSE: Copyright (c) 2025 Yuqun Wu) | Six sequence pairs as examples, re-encoded to 1280 px with metadata stripped; the full dataset is not re-hosted |
| HD-EPIC | [hd-epic.github.io](https://hd-epic.github.io/) | Not stated on its project page; not verified | Source of SceneDiff's kitchen subset only. The six scenes used here are phone captures of rooms, not cooking footage, and are believed to be from the varied subset; this was not confirmed |
| `desk`, `dining_table` | Recorded by the author on a phone; location and device metadata stripped by `pipeline/clean_video.py` | Same as this repository | Examples |

### SceneDiff licence notice

The SceneDiff example clips are used under the MIT licence the authors published. The notice that licence requires:

> MIT License. Copyright (c) 2025 Yuqun Wu. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

Full text: [github.com/yuqunw/scene_diff/blob/main/LICENSE](https://github.com/yuqunw/scene_diff/blob/main/LICENSE).

## Libraries

| Library | Licence |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| NumPy, SciPy | BSD-3-Clause |
| Open3D | MIT |
| OpenCV (`opencv-python`) | Apache-2.0 |
| Transformers, huggingface_hub, safetensors | Apache-2.0 |
| einops | MIT |
| Pillow | HPND |
| imageio-ffmpeg | BSD-2-Clause; downloads an FFmpeg binary (LGPL/GPL build) that is not redistributed here |
| Gradio (demo only) | Apache-2.0 |
| Plotly (demo only) | MIT |

## Citations

```bibtex
@inproceedings{wang2026vggtomega,
  title={VGGT-{$\Omega$}},
  author={Wang, Jianyuan and Chen, Minghao and Zhang, Shangzhan and Karaev, Nikita and Sch{\"o}nberger, Johannes and Labatut, Patrick and Bojanowski, Piotr and Novotny, David and Vedaldi, Andrea and Rupprecht, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

- DINOv3: Siméoni et al., "DINOv3", Meta AI, 2025 ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104)).
- SceneDiff: Wu et al., "SceneDiff: A Benchmark and Method for Multiview Object Change Detection" ([arXiv:2512.16908](https://arxiv.org/abs/2512.16908)).
