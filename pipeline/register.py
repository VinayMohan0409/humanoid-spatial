"""Registration stage: aligns the after cloud to the before cloud with point-to-plane ICP checks that the correction is small, and saves before (pale green), unaligned after (pale red) and aligned after (pale blue) for inspection."""

import json

import numpy as np
import open3d as o3d

from config import ALIGNMENT_DIR, ICP_MAX_DISTANCE, ICP_MAX_ITER, ICP_MAX_ROTATION, ICP_MAX_SHIFT, KEEP_FRACTIONS
from preprocess import AFTER_COLOR, BEFORE_COLOR, preprocess_point_cloud
from tsdf import cloud_path

ALIGNED_COLOR = (0.60, 0.75, 0.95)


def register_point_clouds(source, target, max_distance=ICP_MAX_DISTANCE, max_iter=ICP_MAX_ITER):
    """Register source to target with point-to-plane ICP; return the aligned source and the ICP result."""
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iter
    )
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_distance,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )
    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(result.transformation)
    return aligned, result


def export_registration(keep=KEEP_FRACTIONS[0]):
    """Align after to before, print how well it fit and save the clouds and the transform to output/alignment/."""
    before = preprocess_point_cloud(o3d.io.read_point_cloud(str(cloud_path("before", keep))))
    after = preprocess_point_cloud(o3d.io.read_point_cloud(str(cloud_path("after", keep))))
    aligned, result = register_point_clouds(after, before)

    T = result.transformation
    angle = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    shift = float(np.linalg.norm(T[:3, 3]))
    print(f"fitness {result.fitness:.3f}  inlier RMSE {result.inlier_rmse:.5f}  shift {shift:.4f}  rotation {angle:.2f} deg")

    ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cloud, color in (("before", before, BEFORE_COLOR), ("after", after, AFTER_COLOR), ("after_aligned", aligned, ALIGNED_COLOR)):
        cloud.paint_uniform_color(color)
        o3d.io.write_point_cloud(str(ALIGNMENT_DIR / f"{name}.ply"), cloud)
    np.save(ALIGNMENT_DIR / "after_to_before.npy", T)

    ok = shift <= ICP_MAX_SHIFT and angle <= ICP_MAX_ROTATION
    (ALIGNMENT_DIR / "registration.json").write_text(json.dumps({"fitness": result.fitness, "inlier_rmse": result.inlier_rmse, "shift": shift, "rotation_deg": float(angle),
                                                                  "max_shift": ICP_MAX_SHIFT, "max_rotation_deg": ICP_MAX_ROTATION, "passed": bool(ok)}, indent=2))
    if not ok:  # the videos are reconstructed together, so ICP should only polish; a big correction means they did not land in one coordinate frame
        raise RuntimeError(f"alignment failed: ICP moved the after cloud by {shift:.4f} (limit {ICP_MAX_SHIFT:.4f}) and rotated it {angle:.2f} deg (limit {ICP_MAX_ROTATION} deg), "
                           "so the before and after reconstructions probably did not share a coordinate frame and every change found would be unreliable")
