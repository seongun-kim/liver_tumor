import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
# import torchvision.transforms.v2 as transforms

from models import ResNet50
from datasets import LiverTumorDataset

import utils
import pdb

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='The directory where the original LiTS dataset is saved.')
parser.add_argument('--model_dir', type=str, help='The directory where the model is saved.')
parser.add_argument('--batch_size', type=int)

args = parser.parse_args()


if __name__ == '__main__':
    device = 'cuda'

    model = ResNet50(in_channels=1, num_classes=2)
    checkpoint = torch.load(args.model_dir)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model = torch.nn.DataParallel(model)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

    dataset = LiverTumorDataset(data_dir=args.data_dir, transform=transforms)
    img_to_np = lambda img: img.permute(1, 2, 0).detach().numpy()

    def denormalize_sample(inputs, mean, std):
        return img_to_np(
            inputs
            * torch.tensor(std)
            + torch.tensor(mean)
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Train
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device).float(), label.to(device)
            print(f'batch_idx: {batch_idx}')
            pred = model(data)

            loss = F.nll_loss(pred, label, reduction='sum')
            test_loss += loss

            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(label).sum().item()

    print(f'loss: {loss}')
    print(f'accuracy: {correct / len(dataset)}')
