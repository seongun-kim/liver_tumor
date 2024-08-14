'''
python -m scripts.window_dataset --data_dir /data2/seongun/data/liver_tumor/datasets/L00_T20_W_V01
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

class WindowedTransform:
    def __init__(self, w=0.8, l=0.2):
        self.w = w
        self.l = l

    def __call__(self, tensor):
        px = tensor.clone()
        px_min = self.l - self.w // 2
        px_max = self.l + self.w // 2
        px[px < px_min] = px_min
        px[px > px_max] = px_max
        return (px - px_min) / (px_max - px_min)

# Transforms
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=False),
    # WindowedTransform(w=0.6, l=0.4),
    # transforms.Normalize(mean=[.5], std=[.5]),
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
save_dir = os.path.join(args.data_dir, 'samples')
os.makedirs(save_dir, exist_ok=True)
# for idx in tqdm(range(0, 500, 10), desc='Save sample image'):
for idx in tqdm(range(0, len(dataset), 10), desc='Save sample image'):
    sample, w_sample, label, mask = dataset[idx]
    sample = sample.cpu().numpy()[0]
    w_sample = w_sample.cpu().numpy()[0]
    # sample = denormalize_sample(sample.cpu(), [.5], [.5])
    label = label.item()

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(sample, cmap='gray')
    axs[1].imshow(w_sample, cmap='gray')
    axs[2].imshow(mask, cmap='gray')
    [ax.axis('off') for ax in axs]
    plt.savefig(os.path.join(save_dir, f'i{idx}_l{label}'), bbox_inches='tight')
    plt.close()


# # Define the transformation
# transform = T.Compose([
#     T.Pad((100, 100, 100, 100)),  # Adjust padding as needed
#     T.RandomRotation(45, expand=True)
# ])

# # Example usage
# image = Image.open('path_to_your_image.jpg')
# transformed_image = transform(image)
# transformed_image.show()