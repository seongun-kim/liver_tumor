'''
python -m scripts.prune_dataset \
    --data_dir /data2/seongun/data/liver_tumor/datasets/L00_T20_W \
    --normal_ratio 0.7
'''
import os
import argparse
import numpy as np

import pdb


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='The directory where the original LiTS dataset is saved.')
parser.add_argument('--normal_ratio', type=float, default=0.7, help='The ratio of normal samples to keep.')

args = parser.parse_args()

# Dataset
samples = np.load(os.path.join(args.data_dir, 'samples.npz'))['samples']
w_samples = np.load(os.path.join(args.data_dir, 'windowed_samples.npz'))['windowed_samples']
labels = np.load(os.path.join(args.data_dir, 'labels.npz'))['labels']
masks = np.load(os.path.join(args.data_dir, 'masks.npz'))['masks']

# Indices of normal and tumor samples
normal_indices = np.where(labels == 0)[0]
tumor_indices = np.where(labels == 1)[0]

# Calculate the number of normal samples to keep
# num_normal_to_keep = int(len(normal_indices) * args.normal_ratio)
num_normal_to_keep = int(len(tumor_indices) / (1 - args.normal_ratio) * args.normal_ratio)

# Randomly select normal samples to keep
np.random.seed(0)  # For reproducibility
selected_normal_indices = np.random.choice(normal_indices, num_normal_to_keep, replace=False)

# Combine the selected normal indices with all tumor indices
selected_indices = np.sort(np.concatenate((selected_normal_indices, tumor_indices)))

# Subset the dataset
reduced_samples = samples[selected_indices]
reduced_w_samples = w_samples[selected_indices]
reduced_labels = labels[selected_indices]
reduced_masks = masks[selected_indices]

pdb.set_trace()

# Save the new dataset
np.savez_compressed(os.path.join(args.data_dir, 'reduced_samples_v3.npz'), samples=reduced_samples)
np.savez_compressed(os.path.join(args.data_dir, 'reduced_w_samples_v3.npz'), windowed_samples=reduced_w_samples)
np.savez_compressed(os.path.join(args.data_dir, 'reduced_labels_v3.npz'), labels=reduced_labels)
np.savez_compressed(os.path.join(args.data_dir, 'reduced_masks_v3.npz'), masks=reduced_masks)

print(f"Reduced dataset saved with {len(reduced_samples)} samples from {len(samples)} samples.")
