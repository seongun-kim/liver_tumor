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

        '''
        npz_file = np.load(data_dir)
        # self.samples = np.expand_dims(npz_file['samples'], axis=1)  # (N, 1, W, H)
        self.samples = npz_file['samples']  # (N, W, H)
        self.labels = npz_file['labels']    # (N,)
        self.labels = torch.from_numpy(self.labels).to(torch.int64)
        self.labels = F.one_hot(self.labels)   # (N, 2)
        '''

        self.samples = np.load(os.path.join(data_dir, 'samples.npz'))['samples']
        self.labels = np.load(os.path.join(data_dir, 'labels.npz'))['labels']
        self.labels = torch.from_numpy(self.labels).to(torch.int64)

        # import pdb; pdb.set_trace()

        # self.labels = F.one_hot(self.labels)
        # self.masks = np.load(os.path.join(data_dir, 'masks.npz'))['masks']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.from_numpy(sample).float()

        return sample, label
