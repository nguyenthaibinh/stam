from pathlib import Path
import numpy as np
import pandas as pd
from utils.utils import ConfigLoader
from timeit import default_timer as timer

joint2index = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17}

index2cols = {0: 'x', 1: 'y', 2: 'velocity_x', 3: 'velocity_x', 4: 'acceleration_x', 5: 'acceleration_y',
              6: 'displacement'}

def xdf2tensor(xdf):
    df = xdf.dropna()
    df = df.sort_values(by=['bp', 'frame'], ascending=[1, 1])
    max_frame = df.frame.max()
    min_frame = df.frame.min()
    n_frame = int(max_frame - min_frame + 1)

    data = np.zeros(shape=(7, n_frame, 18))

    for c in index2cols.keys():
        c_name = index2cols[c]
        for joint in joint2index:
            joint_id = joint2index[joint]
            try:
                joint_df = df[df.bp == joint].sort_values(by=['frame'], ascending=[1])
                joint_data = np.asarray(joint_df[c_name])
                len_joint_data = len(joint_data)
                data[c, :len_joint_data, joint_id] = joint_data[:]
            except Exception as e:
                print(e)
                raise e

    return data

def convert_pkl_pose2npy(src_infant_dir, dst_infant_dir):
    pose_file_path = Path(src_infant_dir, 'processed_pose_estimates_coords.pkl')
    xdf = pd.read_pickle(pose_file_path)
    try:
        npy_data = xdf2tensor(xdf)
    except Exception as e:
        raise e
    dst_infant_dir.mkdir(parents=True, exist_ok=True)
    dst_file_path = Path(dst_infant_dir, 'raw_skel.npy')
    if dst_file_path.exists():
        dst_file_path.unlink()
    np.save(dst_file_path, npy_data)
    return npy_data

def main():
    conf = ConfigLoader().config
    data_root = Path(conf['data_root'])

    print(f"Start convert pose to npy data. Data root: {data_root}.")

    src_data_path = Path(data_root, 'interim')
    dst_data_path = Path(data_root, 'cooked')

    src_infant_dirs = [e for e in Path(src_data_path).rglob('*') if e.is_dir()]
    src_infant_dirs = sorted(src_infant_dirs)

    for i, src_infant_dir in enumerate(src_infant_dirs):
        infant_id = src_infant_dir.stem
        dst_infant_dir = Path(dst_data_path, infant_id)
        try:
            start_time = timer()
            convert_pkl_pose2npy(src_infant_dir, dst_infant_dir)
            end_time = timer()
            elapsed_time = end_time - start_time
            print(f"({i + 1:03}/{len(src_infant_dirs):03}) {src_infant_dir} done! "
                  f"Elapsed time: {elapsed_time:.1f} sec.")
        except Exception as e:
            print(e)
            continue

if __name__ == '__main__':
    main()