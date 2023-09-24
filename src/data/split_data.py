from pathlib import Path
import numpy as np
import argparse
from data.dataset import load_labels
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from utils.utils import ConfigLoader

def write_split_data(split_file, X, y):
    with open(split_file, 'w') as f:
        f.write('infant_id,label\n')
        for i, infant_id in enumerate(X):
            label = y[i]
            f.write(f'{infant_id},{label}\n')


def stratified_split(X, y, random_state, n_splits):
    df = pd.DataFrame()
    df['infant_id'] = X
    df['label'] = y
    df = df.sort_values(by='infant_id', ascending=1)
    X_data = np.asarray(df['infant_id'])
    y_label = np.asarray(df['label'])
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=0.2, random_state=random_state
    )
    return sss.split(X_data, y_label)


def split_data(target_dir, X, y, random_state, n_splits=5):
    df = pd.DataFrame()
    df['infant_id'] = X
    df['label'] = y
    df = df.sort_values(by='infant_id', ascending=1)
    X_data = np.asarray(df['infant_id'])
    y_label = np.asarray(df['label'])
    train_test_splits = stratified_split(
        X_data, y_label, random_state=random_state, n_splits=n_splits)
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(train_test_splits):
        # split the whole data to train_val and test
        X_train_val, X_test = X_data[train_val_idx], X_data[test_idx]
        y_train_val, y_test = y_label[train_val_idx], y_label[test_idx]
        
        # split the train_val to train and val set (one split)
        train_val_splits = stratified_split(
            X_train_val, y_train_val, random_state=random_state, n_splits=1)
        
        for train_idx, val_idx in train_val_splits:
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        train_file_path = Path(target_dir, f'train_{fold_idx:02}.txt')
        val_file_path = Path(target_dir, f'val_{fold_idx:02}.txt')
        test_file_path = Path(target_dir, f'test_{fold_idx:02}.txt')
        write_split_data(
            split_file=train_file_path, X=X_train, y=y_train
        )
        write_split_data(
            split_file=val_file_path, X=X_val, y=y_val
        )
        write_split_data(
            split_file=test_file_path, X=X_test, y=y_test
        )

def load_data(data_root, label_dict):
    data_paths = Path(data_root).rglob('pose_sequence.npy')

    infant_list = []
    labels = []

    for data_path in data_paths:
        if not data_path.exists():
            continue
        data_dir = data_path.resolve().parents[0]
        dir_name = str(data_dir.stem)
        infant_id = dir_name[:13]
        infant_list.append(dir_name)
        if infant_id in label_dict.keys():
            labels.append(label_dict[infant_id])
        else:
            labels.append(0)
    return infant_list, labels

def main():
    parser = argparse.ArgumentParser(description="Parse the parameters for the datasets split.")
    parser.add_argument('--n-splits', type=int, default=5, help='Number of splits.')
    parser.add_argument('--random-state', type=int, default=0, help='Random state.')
    args = parser.parse_args()
    n_splits = args.n_splits
    random_state = args.random_state

    conf = ConfigLoader().config
    data_root = Path(conf['data_root'])
    label_file_path = Path(data_root, 'labels.txt')
    split_dir = Path(conf['data_root'], f'folds/random_split')
    split_dir.mkdir(parents=True, exist_ok=True)

    label_dict = load_labels(label_file_path)

    infant_list, labels = load_data(data_root, label_dict)
    split_data(
        split_dir, infant_list, labels,
        random_state=random_state, n_splits=n_splits
    )

if __name__ == '__main__':
    main()
