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
parser.add_argument('--save_dir', type=str, help='The directory where the generated dataset will be saved.')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_epochs', type=int)

args = parser.parse_args()


if __name__ == '__main__':
    resume = False
    device = 'cuda'
    os.makedirs(args.save_dir, exist_ok=True)

    model = ResNet50(in_channels=1, num_classes=2)
    # checkpoint = torch.load('models/pretrained/resnet50_224_1.pth')['net']
    checkpoint = torch.load(args.model_dir)['net']

    if not resume:
        del checkpoint['conv1.weight']
        del checkpoint['fc.weight']
        del checkpoint['fc.bias']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model = torch.nn.DataParallel(model)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

    # dataset = LiverTumorDataset(data_dir='./datasets/L03_T05.npz', transform=transforms)
    dataset = LiverTumorDataset(data_dir=args.data_dir, transform=transforms)
    # dataset = LiverTumorDataset(data_dir='./datasets/L03_T05_subset.npz', transform=transforms)

    # pdb.set_trace()

    img_to_np = lambda img: img.permute(1, 2, 0).detach().numpy()
    def denormalize_sample(inputs, mean, std):
        return img_to_np(
            inputs
            * torch.tensor(std)
            + torch.tensor(mean)
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-4,
    )

    # Train
    for epoch in range(args.num_epochs):
        model.train()
        best_loss = 1e9
        train_loss = 0
        correct = 0
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device).float(), label.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = F.nll_loss(pred, label, reduction='sum')
            loss.backward()
            optimizer.step()
            train_loss += loss
            correct += pred.argmax(dim=1).eq(label).sum().item()
        print(f'epoch: {epoch} - loss: {loss} - accuracy: {correct / len(dataset)}')

        if ((epoch + 1) % 5) == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch:02d}.pt'))
        if train_loss <= best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model.pt'))
