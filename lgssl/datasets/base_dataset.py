import json
import os
from pathlib import Path

import imagesize
import kornia.augmentation as k_transforms
import torch
import torchvision.transforms as tv_transforms
from torchvision.io import ImageReadMode, read_image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.data_root = Path(__file__).parent / "../../data/datasets"

    def get_augmentation(self, name):
        if name == "global_crop":
            augment = torch.nn.Sequential(
                tv_transforms.RandomResizedCrop(224, scale=(0.5, 1.0))
            )
            gpu_augment = torch.nn.Sequential(
                tv_transforms.ConvertImageDtype(torch.float32),
                k_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            )
        elif name == "simclr":
            augment = torch.nn.Sequential(
                tv_transforms.RandomResizedCrop(224, scale=(0.2, 1.0))
            )
            gpu_augment = torch.nn.Sequential(
                tv_transforms.ConvertImageDtype(torch.float32),
                k_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                k_transforms.RandomGrayscale(p=0.2),
                k_transforms.RandomGaussianBlur((15, 15), sigma=(0.1, 2.0), p=0.5),
                k_transforms.RandomHorizontalFlip(),
                k_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            )
        else:
            raise ValueError("Unknown name")

        return augment, gpu_augment

    def __len__(self):
        return len(self.instances)

    def load_image(self, rel_path):
        data_name = "redcaps" if "redcaps" in self.name else self.name
        full_path = str((self.data_root / f"{data_name}/{rel_path}").resolve())

        if "redcaps" in self.name:
            # ignore any jpeg image larger than 0.5mb
            img_size = os.path.getsize(full_path) / 1024
            if img_size > 1024:
                print(f"Skip {full_path}. {img_size:.2f}kbs. decode large jpg -> OOM")
                return None

        # ignore any image with an aspect ratio larger than 10
        W, H = imagesize.get(full_path)
        aspect = max(H, W) / max(min(H, W), 1)
        if aspect > 20:
            print(f"Skip {full_path}. Aspect-ratio {aspect} > 20.")
            return None

        try:
            image = read_image(full_path, ImageReadMode.RGB)
        except:
            print(f"{full_path} is empty")
            return None

        return image

    def get_dict_instances(self):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # load data file and nn file
        LGSSL_ROOT = Path(__file__).parent.parent
        data_file = LGSSL_ROOT / f"../data/data_dicts/{self.name}.json"
        data_dict = json.load(data_file.open())

        # get dictionary
        for tar_id in data_dict:
            for img_id in data_dict[tar_id]:
                path, caption = data_dict[tar_id][img_id][0:2]
                instances.append((tar_id, img_id, path, caption))

        return instances
