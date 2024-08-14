'''
python -m scripts.test_transform --data_dir /data2/seongun/data/liver_tumor/datasets/L00_T20_V01
'''
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from datasets import LiverTumorDataset

import pdb


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='The directory where the original LiTS dataset is saved.')

args = parser.parse_args()

# Config
height, width = 512, 512
diagonal = int((height ** 2 + width ** 2) ** 0.5)
padding = (diagonal - height) // 2, (diagonal - width) // 2

# Transforms
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.Pad(padding),
    transforms.RandomRotation(90, expand=False),
    transforms.Resize((224, 224), antialias=False),
    transforms.Normalize(mean=[.5], std=[.5]),
])

# Dataset
dataset = LiverTumorDataset(data_dir=args.data_dir, transform=transforms)

# Denormalize
sample_to_np = lambda sample: sample.permute(1, 2, 0).squeeze(dim=-1).detach().numpy()
def denormalize_sample(inputs, mean, std):
    return sample_to_np(
        inputs
        * torch.tensor(std)
        + torch.tensor(mean)
    )

# Savefig
save_dir = os.path.join(args.data_dir, 'rotated_samples')
os.makedirs(save_dir, exist_ok=True)
for idx in tqdm(range(0, len(dataset), 50), desc='Save sample image'):
    sample, label, mask = dataset[idx]
    sample = denormalize_sample(sample.cpu(), [.5], [.5])
    label = label.item()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(sample, cmap='gray')
    axs[1].imshow(mask, cmap='gray')
    [ax.axis('off') for ax in axs]
    plt.savefig(os.path.join(save_dir, f'i{idx}_l{label}'), bbox_inches='tight')
    plt.close()
