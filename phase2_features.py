import os
import numpy as np
import pandas as pd
import dataio.process_skeleton as sk
from configs import config_seamer as cf

LEFT_JOINTS = [
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
]

RIGHT_JOINTS = [
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HipRight",
    "KneeRight",
    "AnkleRight",
]


def _extract_3d_array_and_joint_index(full_path: str):
    """Load skeleton frames via repo's SkeletonUtils.load_skeleton (100% unlabeled)."""
    data_sk, _ = sk.sk.load_skeleton(full_path)
    if not data_sk:
        return None, {}

    # Preserve joint ordering from first frame
    first_keys = list(data_sk[0].keys())
    joint_to_idx = {name: i for i, name in enumerate(first_keys)}

    n_frames = len(data_sk)
    n_joints = len(first_keys)
    coords_3d = np.zeros((n_frames, n_joints, 3), dtype=float)

    for f_idx, frame_dict in enumerate(data_sk):
        for j_idx, j_name in enumerate(first_keys):
            if j_name in frame_dict:
                coords_3d[f_idx, j_idx, 0] = frame_dict[j_name][0]
                coords_3d[f_idx, j_idx, 1] = frame_dict[j_name][1]
                coords_3d[f_idx, j_idx, 2] = frame_dict[j_name][2]

    return coords_3d, joint_to_idx


def _compute_bilateral_corr(coords_3d: np.ndarray, joint_to_idx: dict) -> float:
    """Compute Pearson correlation between anatomical Left and Right joint 3D motion series."""
    if coords_3d.shape[0] < 3:
        return 0.0

    left_indices = [joint_to_idx[j] for j in LEFT_JOINTS if j in joint_to_idx]
    right_indices = [joint_to_idx[j] for j in RIGHT_JOINTS if j in joint_to_idx]

    if not left_indices or not right_indices:
        return 0.0

    # Frame-to-frame 3D motion magnitude per joint: shape (n_frames - 1, n_joints)
    step_motion = np.linalg.norm(np.diff(coords_3d, axis=0), axis=2)
    left_series = np.mean(step_motion[:, left_indices], axis=1)
    right_series = np.mean(step_motion[:, right_indices], axis=1)

    if np.std(left_series) < 1e-12 or np.std(right_series) < 1e-12:
        return 0.0

    corr = float(np.corrcoef(left_series, right_series)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def build_phase2_features(
    dataset_path: str = None,
    output_csv: str = "phase2_kinematic_features.csv",
) -> pd.DataFrame:
    target_path = dataset_path if dataset_path else cf.path_dataset
    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError(f"Dataset path not found: {target_path}")

    files = sk.io.get_files(target_path, "(.*txt$)", full_path=False)
    records = []
    rep_counter = 0

    for f in files:
        full_name = os.path.join(target_path, f)
        # Parse filename only to honor dataset's ignored class ('3') like process_files;
        # no labels are stored or used as ML features.
        g_info = sk.split_file_name(f)
        if str(g_info.correct) == "3":
            continue

        coords_3d, joint_to_idx = _extract_3d_array_and_joint_index(full_name)
        if coords_3d is None or coords_3d.shape[0] < 10 or coords_3d.shape[1] < 3:
            continue

        rep_counter += 1
        n_frames, _, _ = coords_3d.shape

        # 1. 3D Spatial Range of Motion (mean 3D span across joints)
        ptp_3d = np.ptp(coords_3d, axis=0)  # (n_joints, 3)
        joint_span_3d = np.linalg.norm(ptp_3d, axis=1)
        spatial_rom_3d = float(np.mean(joint_span_3d))

        # 2. Frame-to-frame 3D Speed (displacement per frame)
        step_disp = np.diff(coords_3d, axis=0)  # (n_frames-1, n_joints, 3)
        speed_3d = np.linalg.norm(step_disp, axis=2)  # (n_frames-1, n_joints)
        mean_speed_3d = float(np.mean(speed_3d))
        p95_speed_3d = float(np.percentile(speed_3d, 95))

        # 3. 3D Acceleration magnitude (2nd difference across frames)
        if n_frames >= 3:
            accel_3d = np.linalg.norm(np.diff(coords_3d, n=2, axis=0), axis=2)
            mean_accel_3d = float(np.mean(accel_3d))
        else:
            mean_accel_3d = 0.0

        # 4. 3D Jerk / Smoothness magnitude (3rd difference across frames)
        if n_frames >= 4:
            jerk_3d = np.linalg.norm(np.diff(coords_3d, n=3, axis=0), axis=2)
            mean_jerk_3d = float(np.mean(jerk_3d))
        else:
            mean_jerk_3d = 0.0

        # 5. Normalized Movement Variability (joint std normalized by spatial RoM)
        joint_std_3d = np.mean(np.linalg.norm(np.std(coords_3d, axis=0), axis=1))
        norm_variability = float(joint_std_3d / (spatial_rom_3d + 1e-6))

        # 6. Anatomical Left-Right motion correlation
        bilateral_corr = _compute_bilateral_corr(coords_3d, joint_to_idx)

        records.append({
            "Rep_ID": f"REP_{rep_counter:04d}",
            "n_frames": int(n_frames),
            "spatial_rom_3d": round(spatial_rom_3d, 5),
            "mean_speed_3d": round(mean_speed_3d, 5),
            "p95_speed_3d": round(p95_speed_3d, 5),
            "mean_accel_3d": round(mean_accel_3d, 5),
            "mean_jerk_3d": round(mean_jerk_3d, 5),
            "norm_variability": round(norm_variability, 5),
            "bilateral_motion_corr": round(bilateral_corr, 5),
        })

    if not records:
        raise ValueError("No valid skeleton sequences found.")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    out_df = build_phase2_features()
    print(f"Phase 2 complete: {len(out_df)} repetitions -> phase2_kinematic_features.csv")
    print(out_df.head(3).to_string(index=False))