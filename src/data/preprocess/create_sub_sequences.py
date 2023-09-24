import numpy as np
from pathlib import Path
import argparse
from utils.utils import ConfigLoader
from timeit import default_timer as timer

def break_down_one_pose(src_infant_dir, dst_data_root, block_size=1000, stride=500):
    infant_id = src_infant_dir.stem
    src_data_file = Path(src_infant_dir, 'pose_sequence.npy')
    data = np.load(src_data_file)
    length = data.shape[1]
    for i, start in enumerate(range(0, length, stride)):
        end = start + block_size
        if end > length:
            break
        sub_data = data[:, start:end, :]
        dst_infant_dir = Path(dst_data_root, f'{infant_id}_sub_{i + 1}')
        dst_infant_dir.mkdir(parents=True, exist_ok=True)
        dst_file_path = Path(dst_infant_dir, f'pose_sequence.npy')
        np.save(dst_file_path, sub_data)

def break_down_poses(block_size, overlapping_size):
    conf = ConfigLoader().config
    data_root = conf['data_root']

    stride = block_size - overlapping_size

    print(f"Start breaking to sub sequences. Data root: {data_root}.")

    src_data_root = Path(data_root, f'cooked')
    dst_data_root = Path(data_root, f'cooked_subsequences_{block_size:04}_{overlapping_size:04}')
    src_infant_dirs = [e for e in Path(src_data_root).rglob('*') if e.is_dir()]
    src_infant_dirs = sorted(src_infant_dirs)

    num_infants = len(src_infant_dirs)
    for i, src_infant_dir in enumerate(src_infant_dirs):
        start_time = timer()
        break_down_one_pose(src_infant_dir, dst_data_root, block_size, stride)
        end_time = timer()
        elapsed_time = end_time - start_time
        print(f"{i:03}/{num_infants} {src_infant_dir} done!. Elapsed time: {elapsed_time:.1f} sec.")

def main():
    parser = argparse.ArgumentParser(description="Parse the parameters for breaking down the pose sequences.")
    parser.add_argument('--block-size', type=int, default=1000, help='Length of the sub-sequences.')
    parser.add_argument('--overlapping-size', type=int, default=200,
                        help='The size of the overlapping window (in frame).')
    args = parser.parse_args()
    block_size = args.block_size
    overlapping_size = args.overlapping_size

    break_down_poses(block_size, overlapping_size)

if __name__ == '__main__':
    main()