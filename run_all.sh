# Preprocess pandas files and save to {{ data_root }}/estimated_poses and {{ data_root }}/preprocessed_poses folders
python src/data/preprocess/preprocess_pose_data.py

# Create pose sequences (npy files will be stored in {{ data_root }}/pose_sequences/{{ infant_id }}/raw_skel.npy)
python src/data/preprocess/pkl_pose2npy.py

# Break a long sequences to sub-sequences of length 1000 frames
python src/data/preprocess/create_sub_sequences.py

# Split datasets into folds
python src/data/split_data.py