import os
from pathlib import Path

import torchvision.transforms as transforms
from hydra.utils import instantiate
from PIL import ImageFile
from torch.utils.data import DataLoader, default_collate

from .catalog import DatasetCatalog

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def build_loader(cfg, num_gpus=1, n_workers=None):
    """
    Build a PyTorch dataloader and the underlying dataset (using config).
    """

    # Build a dataset from the provided dataset config.
    dataset = instantiate(cfg)

    if n_workers is None:
        n_workers = len(os.sched_getaffinity(0)) // num_gpus - 1

    loader = DataLoader(
        dataset=dataset,
        batch_size=int(cfg.batch_size / num_gpus),
        shuffle=True,
        pin_memory=True,
        collate_fn=skip_bad_collate,
        num_workers=n_workers,
    )
    return loader


def skip_bad_collate(batch):
    # filter Nones
    b_size = len(batch)
    batch = [x for x in batch if x is not None]
    if len(batch) < b_size:
        print(f"Skipped {len(batch) - b_size} instances because of None batches.")

    output_dict = {}
    for key in batch[0]:
        if "cap_token" in key:
            output_dict[key] = [x[key] for x in batch]
        elif "augmentation" in key:
            # This only works because we use Kornia' augmentations which can handle
            # randomization within batches so you only need one instance of it
            output_dict[key] = batch[0][key]
        else:
            output_dict[key] = default_collate([x[key] for x in batch])

    return output_dict


def get_downstream_dataset(name, split, transform):
    dataset_root = Path(__file__).parent / "../../data/datasets"

    dataset = DatasetCatalog.build_dataset(
        name, root=dataset_root, split=split, transform=transform
    )
    return dataset


def get_linearprobe_loaders(name, image_mean="imagenet"):
    if image_mean == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    elif image_mean == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError()

    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            lambda x: x.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_set = get_downstream_dataset(name, "train", transform)
    valid_set = get_downstream_dataset(name, "val", transform)
    test_set = get_downstream_dataset(name, "test", transform)

    bs = 64
    # TensorFlow datasets sometimes freak out with parallel and just hang.
    # Unclear why but this seems to solve it.
    n_workers = 0  # len(os.sched_getaffinity(0))
    train_loader = DataLoader(train_set, bs, num_workers=n_workers, drop_last=False)
    valid_loader = DataLoader(valid_set, bs, num_workers=n_workers, drop_last=False)
    test_loader = DataLoader(test_set, bs, num_workers=n_workers, drop_last=False)
    return train_loader, valid_loader, test_loader
