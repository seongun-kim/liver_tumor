from typing import Optional, List
import os
import json
import numpy as np
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch import Tensor
from torch.utils.data import Dataset, Subset
from PIL import Image

import requests
import hashlib
import zipfile
from tqdm import tqdm


def get_medmnist_dataset(
        dataset,
        transform,
        subset_size: int = 100,  # ignored if indices is not None
        root_dir="./data/medmnist",
        indices: Optional[List[int]] = None,
        download=False,
):
    os.chdir(Path(__file__).parent)  # ensure path
    dataset = MedMNISTDataset(dataset=dataset, root_dir=root_dir, transform=transform, download=download)
    if indices is not None:
        return Subset(dataset, indices=indices)
    indices = list(range(len(dataset)))
    subset = Subset(dataset, indices=indices[:subset_size])
    return subset


def get_medmnist_model(dataset, num_classes, url=None, md5=None, root_dir="./", target_file="resnet18_224_1.pth", download=False):
    WEIGHT_URL = {"pneumoniamnist": ["https://zenodo.org/records/7782114/files/weights_pneumoniamnist.zip",
                                  "01ee733fc0e3263e0b809e2b0182594d"]}
    
    download_url = [url, md5] if url is not None else WEIGHT_URL[dataset]
    weight_path = os.path.join(root_dir, download_url[0].split("/")[-1])
    model_path = os.path.join(root_dir, target_file)

    if download and not os.path.exists(weight_path):
        download_(url=download_url[0], root_dir=root_dir, expected_md5=download_url[1])
        extract_zip_(zip_path=weight_path)

    model = torchvision.models.resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path +'/' + target_file)["net"])

    transform = transforms.Compose(
        [transforms.ToImage(),
         transforms.RGB(),
         transforms.ToDtype(torch.float32, scale=True),
         transforms.Normalize(mean=[.5], std=[.5])])

    return model, transform

def download_(url, root_dir="./", expected_md5=None):
    # Download the file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    output_path = os.path.join(root_dir, url.split('/')[-1])
    with open(output_path, 'wb') as file, tqdm(
            total=total_size, unit='iB', unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    if expected_md5 is not None:
        # Check the MD5 hash
        md5 = hashlib.md5()
        with open(output_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)

        if md5.hexdigest() != expected_md5:
            raise ValueError("MD5 checksum does not match")
        
def extract_zip_(zip_path, root_dir="./", target_file="resnet18_224_1.pth"):
    # Extract the specific file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(target_file, os.path.join(root_dir, target_file))

