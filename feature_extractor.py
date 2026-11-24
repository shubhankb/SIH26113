import os
import numpy as np
import pandas as pd
import dataio.process_skeleton as sk
from configs import config_seamer as cf


def extract_unsupervised_features(dataset_path=None):
    target_path = dataset_path if dataset_path else cf.path_dataset
    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError(f"Dataset path not found: {target_path}")

    # process_files returns (data, data_info, file_names)
    # Using '_' for data_info so labels/metadata are not used in unsupervised feature extraction
    data, _, _ = sk.process_files(target_path)

    records = []
    for idx, seq in enumerate(data, start=1):
        arr = np.asarray(seq, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 10 or arr.shape[1] < 3:
            continue

        # Each row is a frame; columns are 3D (x, y, z) values across joints (feature_dim = 75 -> 25 joints x 3)
        n_frames, n_cols = arr.shape
        n_joints = n_cols // 3
        coords_3d = arr[:, : n_joints * 3].reshape(n_frames, n_joints, 3)

        # 1. Spatial Range of Motion (mean 3D span across joints)
        joint_spans = np.linalg.norm(np.ptp(coords_3d, axis=0), axis=1)
        rom_3d = float(np.mean(joint_spans))

        # 2. Frame-to-frame 3D displacement / velocity magnitude
        vel_3d = np.linalg.norm(np.diff(coords_3d, axis=0), axis=2)
        mean_vel = float(np.mean(vel_3d))
        max_vel = float(np.percentile(vel_3d, 95))

        # 3. Third-difference smoothness / jerk magnitude (when enough frames exist)
        if n_frames >= 4:
            jerk_3d = np.linalg.norm(np.diff(coords_3d, n=3, axis=0), axis=2)
            movement_jerk = float(np.mean(jerk_3d))
        else:
            movement_jerk = 0.0

        # 4. Left-Right joint group motion correlation
        mid = n_joints // 2
        if mid > 0 and n_frames >= 2:
            left_motion = np.mean(np.linalg.norm(coords_3d[:, :mid, :], axis=2), axis=1)
            right_motion = np.mean(np.linalg.norm(coords_3d[:, mid : 2 * mid, :], axis=2), axis=1)
            if np.std(left_motion) > 0 and np.std(right_motion) > 0:
                corr = float(np.corrcoef(left_motion, right_motion)[0, 1])
                symmetry_idx = 0.0 if np.isnan(corr) else corr
            else:
                symmetry_idx = 0.0
        else:
            symmetry_idx = 0.0

        # 5. Normalized trajectory variability
        variability = float(np.mean(np.std(coords_3d, axis=0)) / (rom_3d + 1e-6))

        records.append({
            "Rep_ID": f"REP_{idx:04d}",
            "Range_of_Motion_deg": round(rom_3d, 4),
            "Mean_Angular_Vel": round(mean_vel, 4),
            "Max_Angular_Vel": round(max_vel, 4),
            "Movement_Jerk": round(movement_jerk, 4),
            "Joint_Symmetry_Index": round(symmetry_idx, 4),
            "Rep_Variability": round(variability, 4),
        })

    if not records:
        raise ValueError("No valid skeleton sequences found in loaded data.")

    df = pd.DataFrame(records)
    df.to_csv("intellirehab_unsupervised_features.csv", index=False)
    return df


if __name__ == "__main__":
    features_df = extract_unsupervised_features()
    print(f"Extracted {len(features_df)} unlabeled movement feature rows -> intellirehab_unsupervised_features.csv")