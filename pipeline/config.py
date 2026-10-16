"""Shared paths and tunable parameters for every pipeline stage."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = Path(os.environ.get("PIPELINE_OUTPUT", ROOT / "output"))  # main.py points these at the pair being processed
REPOS = ROOT / "repos"

VIDEOS = ("before", "after")
VIDEO_FILES = {"before": os.environ.get("PIPELINE_BEFORE", "before.mp4"), "after": os.environ.get("PIPELINE_AFTER", "after.mp4")}

BIN_SECONDS = 0.3
MIN_BINS = 16
MAX_BINS = 30
MIN_GAP_FRAMES = 7

CHECKPOINT_REPO = "facebook/VGGT-Omega"
CHECKPOINT_FILE = "vggt_omega_1b_512.pt"
IMAGE_RESOLUTION = 512

VOXEL_SIZE = float(os.environ.get("PIPELINE_VOXEL", 0.005))  # main.py scales this to the scene after the reconstruction
SDF_TRUNC = 4 * VOXEL_SIZE
WEIGHT_THRESHOLD = 3.0
KEEP_FRACTIONS = (0.97,)
FAR_DEPTH_FACTOR = 3.0  # depth beyond this many times the scene's median depth is ignored: tabletop changes happen near the camera, and far readings are mostly stray points
OUTLIER_NEIGHBOURS = 20
OUTLIER_STD_RATIO = 2.0
ICP_MAX_ITER = 100
ICP_MAX_DISTANCE = 2 * VOXEL_SIZE
ICP_MAX_SHIFT = 5 * VOXEL_SIZE  # the two videos are reconstructed together, so ICP should only polish: a larger shift means the reconstruction failed
ICP_MAX_ROTATION = 3.0  # degrees, same reason
HEATMAP_MAX = 10 * VOXEL_SIZE
SEGMENTS_DIR = OUTPUT / "segments"
MAX_PLANES = 6
MIN_PLANE_FRACTION = 0.05
PLANE_BAND = 1 * VOXEL_SIZE
PLANE_FEATURE_SIM = 0.7
OBJECT_MIN_POINTS = 50
OBJECT_RADIUS = 1.6 * VOXEL_SIZE
OBJECT_LINK_SIM = 0.8
OBJECT_GROW_SIM = 0.85
CHECK_FRAME = 14
PRESENT_RADIUS = 4 * VOXEL_SIZE
SEE_THROUGH_MARGIN = 1 * VOXEL_SIZE
MIN_VIEW_FRACTION = 0.25
MOVE_DISTANCE = 10 * VOXEL_SIZE
APPEARANCE_MAX = 0.6  # feature dissimilarity (1 - cosine) drawn fully red in the appearance map
CHROMA_MAX = 20.0  # Lab a,b distance drawn fully red in the appearance map
Z_STRONG = 3.5  # robust z-score of feature dissimilarity that alone marks an appearance change
Z_WEAK = 2.0  # lower z-score that counts when feature and chroma both reach it
UNIT_FLAG_SHARE = 0.3  # share of a unit's points flagged as changed (or median z above Z_WEAK) for it to count as changed
IDENTITY_MIN = 0.62  # lowest identity score accepted as the same object in both videos (true pairs 0.66+, others 0.58 or less on our three scenes)
DINO_MIN = 0.75  # lowest pooled DINO cosine for two units to be the same object (true pairs 0.86+, a false pair 0.45 on our scenes)
ABSORB_MAX_RATIO = 8  # a change region is absorbed into the segment it lies in only if that segment has at most this many times its points
RING_RADIUS = 6 * VOXEL_SIZE  # a unit's surroundings are the points within this distance of it
RING_MAX = 0.92  # (mean test, kept for comparison)
RING_POINT_SIM = 0.9  # a unit point whose feature has cosine below this to the surroundings' mean is distinct from them
RING_DISTINCT_SHARE = 0.3  # a unit with a smaller share of distinct points is an empty footprint, not an object (the same share as UNIT_FLAG_SHARE)  # a unit whose feature is more similar than this to its surroundings is an empty footprint, not an object
CHROMA_BINS = 12  # bins per axis of the Lab a,b histogram in an object descriptor
SMOOTH_K = 12  # neighbours in the majority filter that removes speckle from the change flags

FRAMES_DIR = OUTPUT / "frames"
RECON_DIR = OUTPUT / "vggt"
TSDF_DIR = OUTPUT / "tsdf"
ALIGNMENT_DIR = OUTPUT / "alignment"
DINO_DIR = OUTPUT / "dino"
FEATURES_DIR = OUTPUT / "features3d"
VISIBILITY_TOL = 0.03
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"

VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"  # small enough for a laptop; Apache-2.0
VLM_PROMPT = "What object is this? Answer with one word. If you don't know, answer unknown."
VLM_VIEWS = 3  # key frames per unit that are shown to the model; the most common answer wins
VLM_MARGIN = 0.3  # context kept around the unit in each crop, as a share of its size
VLM_MIN_VISIBLE = 10  # fewest unit points a frame must show to be used
