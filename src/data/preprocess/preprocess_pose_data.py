from pathlib import Path
import pandas as pd
from data.preprocess.data_utils import get_video_information, load_raw_pkl_files, edit_df, get_skel, \
    interpolate_df, smooth, normalise_skeletons, get_joint_angles, get_dynamics_xy, get_dynamics_angle
from utils.utils import ConfigLoader
from timeit import default_timer as timer

def compute_motion_features(df):
    median_window = 1
    mean_window = 1
    delta_window = .25  # smoothing applied to delta_x, velocity, acceleration

    df['x'] = pd.to_numeric(df['x'])
    df['y'] = pd.to_numeric(df['y'])
    df['x'] = (df['x'] - df['pixel_x'] / 2) / df['pixel_y']
    df['y'] = (df['y'] - df['pixel_y'] / 2) / df['pixel_y']
    # interpolate
    df = df.groupby(['video', 'bp']).apply(interpolate_df).reset_index(drop=True)
    # median and mean filter
    median_window = .5
    mean_window = .5
    df = df.groupby(['video', 'bp']).apply(lambda x: smooth(x, 'y', median_window, mean_window)).reset_index(drop=True)
    df = df.groupby(['video', 'bp']).apply(lambda x: smooth(x, 'x', median_window, mean_window)).reset_index(drop=True)
    # rotate and normalise by reference
    xdf = normalise_skeletons(df)
    # extract angles
    adf = get_joint_angles(df)
    # get dynamics
    xdf = get_dynamics_xy(xdf, delta_window)
    adf = get_joint_angles(xdf)
    adf = get_dynamics_angle(adf, delta_window)
    return xdf, adf

def preprocess_pose(src_infant_dir, dst_infant_dir):
    df_pkl = load_raw_pkl_files(src_infant_dir)
    df_fps = get_video_information(src_infant_dir)
    df = df_pkl.groupby(['video', 'frame']).apply(get_skel)
    df = edit_df(df, df_fps)
    xdf, adf = compute_motion_features(df)

    dst_infant_dir.mkdir(parents=True, exist_ok=True)
    if Path(dst_infant_dir, 'pose_estimates.pkl').exists():
        Path(dst_infant_dir, 'pose_estimates.pkl').unlink()
    if Path(dst_infant_dir, 'processed_pose_estimates_coords.pkl').exists():
        Path(dst_infant_dir, 'processed_pose_estimates_coords.pkl').unlink()
    if Path(dst_infant_dir, 'processed_pose_estimates_angles.pkl').exists():
        Path(dst_infant_dir, 'processed_pose_estimates_angles.pkl').unlink()

    df.to_pickle(Path(dst_infant_dir, 'pose_estimates.pkl'))
    xdf.to_pickle(Path(dst_infant_dir, 'processed_pose_estimates_coords.pkl'))
    adf.to_pickle(Path(dst_infant_dir, 'processed_pose_estimates_angles.pkl'))

def main():
    conf = ConfigLoader().config
    data_root = conf['data_root']
    print(f"Start preprocessing poses. Data root: {data_root}.")
    src_data_path = Path(data_root, 'openpose_estimated')
    dst_data_path = Path(data_root, 'interim')
    src_infant_dirs = [e for e in Path(src_data_path).rglob('*') if e.is_dir()]
    src_infant_dirs = sorted(src_infant_dirs)

    for i, src_infant_dir in enumerate(src_infant_dirs):
        infant_id = str(src_infant_dir.stem)
        dst_infant_dir = Path(dst_data_path, infant_id)
        start_time = timer()
        preprocess_pose(src_infant_dir, dst_infant_dir)
        end_time = timer()
        elapsed_time = end_time - start_time
        print(f"({i + 1:03}/{len(src_infant_dirs):03}): {infant_id} done! Elapsed time: {elapsed_time:.1f} sec.")

if __name__ == '__main__':
    main()