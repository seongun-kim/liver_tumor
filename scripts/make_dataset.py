import os
import glob
import shutil
import argparse
from tqdm import tqdm

import numpy as np 
import nibabel as nib
import matplotlib.pyplot as plt

import pdb

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='The directory where the original LiTS dataset is saved.')
parser.add_argument('--save_dir', type=str, help='The directory where the generated dataset will be saved.')
parser.add_argument('--ratio_liver', type=float, help='The proportion of pixels that represent the liver.')
parser.add_argument('--ratio_tumor', type=float, help='The proportion of pixels that represent the tumor, relative to the liver.')
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()


# Function to read .nii file and return a pixel array
def read_nii(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    scan = np.rot90(np.array(scan)).transpose(2, 0, 1)
    return scan


# Function to normalize pixel values using min-max normalization
def normalize(scan):
    maxs = np.max(scan)
    mins = np.min(scan)
    return (scan - mins) / (maxs - mins)


# Main function
def main(args):
    # Create save directory
    save_dir = f'{args.save_dir}/L{int(args.ratio_liver*100):02d}_T{int(args.ratio_tumor*100):02d}'
    if os.path.exists(save_dir):
        confirm = input(f"'{save_dir}' already exists. Are you sure you want to delete '{save_dir}'? (y/n): ")
        if confirm.lower() == 'y':
            shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Initialize lists to store CT scans and labels
    sample_cts = []
    sample_masks = []
    tumors = []

    # Loop over all volumes
    for volume_idx in tqdm(range(1, 9, 1), desc='Processing volumes', position=0):
        ct_files = glob.glob(os.path.join(args.data_dir, f'volume_pt{volume_idx}/*.nii'))
        ct_files = sorted(ct_files)
        for ct_file in tqdm(ct_files, desc=f'Processing files in volume {volume_idx}', position=1, leave=False):
            nii_idx = int(ct_file.replace('.', '-').split('-')[-2])
            mask_file = os.path.join(args.data_dir, f'segmentations/segmentation-{nii_idx}.nii')

            # Read and normalize CT scan
            sample_ct = normalize(read_nii(ct_file))
            sample_mask = read_nii(mask_file)

            # print(f'\nct-{nii_idx}')
            # print(f'ct: {sample_ct.shape}\nmask: {sample_mask.shape}')
            # print(f'max: {sample_ct.max()}\nmin: {sample_ct.min()}')

            # Loop over all slices in the CT scan
            for slice_idx in tqdm(range(len(sample_ct)), desc='Processing slices in CT scan', position=2, leave=False):
                height, width = sample_mask[slice_idx].shape
                num_pixels = height * width
                num_background = (sample_mask[slice_idx] == 0.).sum()
                num_normal = (sample_mask[slice_idx] == 1.).sum()
                num_tumor = (sample_mask[slice_idx] == 2.).sum()
                num_liver = num_normal + num_tumor

                # Ignore a CT scan sample if there is no liver
                if num_liver == 0:
                    continue

                ratio_liver = num_liver / num_pixels
                ratio_tumor = num_tumor / num_liver

                # Save CT samples if liver takes at least {args.ratio_liver*100}% of pixels
                if ratio_liver >= args.ratio_liver:
                    # Consider a CT sample as a tumor if there exists at least {args.ratio_tumor*100}% of tumor inside the liver
                    if ratio_tumor >= args.ratio_tumor:
                        sample_cts.append(sample_ct[slice_idx])
                        sample_masks.append(sample_mask[slice_idx])
                        tumors.append(1.)
                    # Consider a sample as normal if there is no tumor inside the liver
                    elif num_tumor == 0:
                        sample_cts.append(sample_ct[slice_idx])
                        sample_masks.append(sample_mask[slice_idx])
                        tumors.append(0.)

            # Visualize
            if args.visualize:
                os.makedirs(os.path.join(save_dir, f'nii_{nii_idx:02d}/'), exist_ok=True)
                for slice_idx in range(0, len(sample_ct), 20):
                    sliced_ct = sample_ct[slice_idx, :, :]
                    sliced_mask = sample_mask[slice_idx, :, :]

                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(sliced_ct, cmap='gray')
                    axs[1].imshow(sliced_mask, cmap='gray')

                    [ax.axis('off') for ax in axs]
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'nii_{nii_idx:02d}/slice_{slice_idx:02d}'))
                    plt.close(fig)

    # Check if the number of samples and labels match
    assert len(sample_cts) == len(tumors) and len(sample_cts) == len(sample_masks), \
        'The number of samples and labels mismatches'

    np.savez_compressed(
        os.path.join(save_dir, 'samples'),
        samples=np.array(sample_cts),
    )
    np.savez_compressed(
        os.path.join(save_dir, 'labels'),
        labels=np.array(tumors),
    )
    np.savez_compressed(
        os.path.join(save_dir, 'masks'),
        masks=np.array(sample_masks)
    )


if __name__ == '__main__':
    main(args)