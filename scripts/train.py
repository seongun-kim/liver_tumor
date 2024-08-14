import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms

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
    # Set seed
    seed = 0
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    resume = False
    device = 'cuda'
    os.makedirs(args.save_dir, exist_ok=True)

    # Model
    model = ResNet50(in_channels=1, num_classes=2)
    checkpoint = torch.load(args.model_dir)['net']

    if not resume:
        del checkpoint['conv1.weight']
        del checkpoint['fc.weight']
        del checkpoint['fc.bias']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model = torch.nn.DataParallel(model)

    
    # Dataset
    height, width = 512, 512
    diagonal = int((height ** 2 + width ** 2) ** 0.5)
    padding = (diagonal - height) // 2, (diagonal - width) // 2

    transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Pad(padding),
        transforms.RandomRotation(90, expand=False),
        transforms.Resize((224, 224), antialias=False),
        # transforms.Normalize(mean=[.5], std=[.5]),
    ])

    dataset = LiverTumorDataset(data_dir=args.data_dir, transform=transforms)

    img_to_np = lambda img: img.permute(1, 2, 0).detach().numpy()
    def denormalize_sample(inputs, mean, std):
        return img_to_np(
            inputs
            * torch.tensor(std)
            + torch.tensor(mean)
        )

    num_data = len(dataset)
    num_train = int(num_data * 0.7)
    # num_train = int(num_data * 0.1)
    num_test = num_data - num_train

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4,
    )

    # Train
    for epoch in range(args.num_epochs):
        model.train()
        best_loss = 1e9
        train_loss = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device).float(), label.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = F.nll_loss(pred, label, reduction='mean')
            loss.backward()
            optimizer.step()
            train_loss += loss
        print(f'epoch: {epoch} - loss: {loss}')

        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.to(device).float(), label.to(device)
                pred = model(data)

                loss = F.nll_loss(pred, label, reduction='sum').item()
                pred = pred.argmax(dim=1)
                correct += pred.eq(label).sum().item()

                test_loss += loss

        print(f'epoch: {epoch} - loss: {test_loss} - accuracy: {correct / len(test_loader.dataset)}\n')

        if ((epoch + 1) % 10) == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch:02d}.pt'))
        if train_loss <= best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model.pt'))
