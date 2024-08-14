import os
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms


class LiverTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform

        # self.samples = np.load(os.path.join(data_dir, 'reduced_samples_v3.npz'))['samples']
        self.w_samples = np.load(os.path.join(data_dir, 'reduced_w_samples_v3.npz'))['windowed_samples']
        self.labels = np.load(os.path.join(data_dir, 'reduced_labels_v3.npz'))['labels']
        self.labels = torch.from_numpy(self.labels).to(torch.int64)

        # self.masks = np.load(os.path.join(data_dir, 'reduced_masks_v3.npz'))['masks']

    def highlight(self, samples):
        samples = samples * 2 - 1.
        samples = np.abs(samples)
        samples = samples - 1.
        samples = samples * -1
        return samples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # sample = self.samples[idx]
        w_sample = self.w_samples[idx]
        label = self.labels[idx]
        # mask = self.masks[idx]

        if self.transform:
            # sample = self.transform(sample)
            w_sample = self.transform(w_sample)
        else:
            # sample = torch.from_numpy(sample).float()
            w_sample = torch.from_numpy(w_sample).float()

        # return sample, label
        return w_sample, label
