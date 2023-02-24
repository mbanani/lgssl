from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import tensorflow_datasets as tfds
import torch
import torchvision.datasets as TD
from PIL import Image
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import ShardingFilter
from torchvision.datasets import ImageFolder

from lgssl.utils.metrics import Accuracy


class TfdsWrapper(IterDataPipe):
    """
    Minimal wrapper on ``tensorflow-datasets`` to serve ``(image, label)``
    tuples for image classification datasets, similar to torchvision.
    """

    def __init__(
        self, name: str, root: str | Path, split: str, transform: Callable | None = None
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.transform = transform

        dset = tfds.load(name, split=split, data_dir=root)
        dset = tfds.as_numpy(dset)

        # Record length of the dataset before further wrapping.
        self._length = len(dset)

        # Wrap the tensorflow dataset with `IterDataPipe` and apply sharding filter
        # to avoid duplicates when multiple CPU workers are used in DataLoader.
        self.dset = ShardingFilter(dset)

    def __repr__(self):
        return f"TfDatasetWrapper(name={self.name}, split={self.split})"

    def __len__(self):
        return self._length

    def __iter__(self) -> Iterator[tuple[Image.Image, torch.Tensor]]:
        for instance in self.dset:
            # Convert numpy arrays: image (PIL.Image) and label (tensor).
            # Handle special case with MNIST images.
            if self.name == "mnist":
                image = Image.fromarray(instance["image"][..., 0], mode="L")
            else:
                image = Image.fromarray(instance["image"])

            image = image.convert("RGB")
            label = torch.tensor(instance["label"])

            if self.transform is not None:
                image = self.transform(image)

            yield image, label


class CLEVRCounts(TfdsWrapper):
    """
    CLEVR-Counts image classification dataset. COunting the number of objects in
    a scene is framed as a classification task. This task was included in the
    Visual Task Adaptation Benchmark (VTAB), and used in CLIP evaluation suite.
    """

    def __init__(self, root: str | Path, split: str, transform: Callable | None = None):
        super().__init__("clevr", root, split, transform)

        # Convert counts to contiguous labels.
        self._labels = [10, 3, 4, 5, 6, 7, 8, 9]

    def __iter__(self) -> Iterator[tuple[Image.Image, torch.Tensor]]:
        for instance in self.dset:
            image = Image.fromarray(instance["image"]).convert("RGB")
            num_objects = len(instance["objects"]["color"])
            label = torch.tensor(self._labels.index(num_objects))

            if self.transform is not None:
                image = self.transform(image)

            yield image, label


class ImageNet(ImageFolder):
    def __init__(self, root: str, split: str = "train", **kwargs):
        super().__init__(str(Path(root) / split), **kwargs)


class DatasetCatalog:
    """
    A catalog and constructor for all supported evaluation datasets in the package.
    This class holds essential information for each dataset in class attributes,
    and member functions (:meth:`build_dataset` and :meth:`build_metric`) use this
    information to create datasets.

    Right now the catalog contains all datasets of Kornblith-12 w/o VOC.

    Attributes:
        CONSTRUCTORS: Dictionary mapping between dataset name and a partial
            callable that shows which implementation we are using. Current
            datasets are either sourced from ``tensorflow-datasets`` library
            or are extensions of implementations in ``torchvision.datasets``.

        SPLITS: Dictionary mapping between dataset name and a list of three
            strings that contain names of ``[train, val, test]`` splits in that
            order. For datasets that do not have an official val split, we use
            splits notation of Tensorflow datasets to deterministically hold out
            a random 10% subset of train split.

        METRICS: Dictionary mapping between dataset name and a partial callable
            that can create the suitable metric for the dataset.

    Examples:
        >>> dset = DatasetCatalog.build_dataset(
                "food101",
                root="datasets/eval",
                split="test",
                transform=T.Compose(
                    [T.Resize(224), T.CenterCrop(224), T.ToTensor()]
                )
            )
        >>> # Can wrap this into dataloader as usual PyTorch datasets.
        >>> dataloader = DataLoader(dset, batch_size=128, num_workers=10)
        >>> for images, labels in dataloader:
        >>>     pass  # Run evaluation.
    """

    CONSTRUCTORS: dict[str, Callable] = {
        "food101": partial(TfdsWrapper, name="food101"),
        "cifar10": partial(TfdsWrapper, name="cifar10"),
        "cifar100": partial(TfdsWrapper, name="cifar100"),
        "cub2011": partial(TfdsWrapper, name="caltech_birds2011"),
        "sun397": partial(TfdsWrapper, name="sun397/standard-part1-120k"),
        "cars196": partial(TfdsWrapper, name="cars196"),
        "aircraft": partial(TD.FGVCAircraft, download=True),
        "dtd": partial(TfdsWrapper, name="dtd"),
        "pets": partial(TfdsWrapper, name="oxford_iiit_pet"),
        "caltech101": partial(TfdsWrapper, name="caltech101"),
        "flowers": partial(TfdsWrapper, name="oxford_flowers102"),
        "stl10": partial(TfdsWrapper, name="stl10"),
        "eurosat": partial(TfdsWrapper, name="eurosat"),
        "resisc45": partial(TfdsWrapper, name="resisc45"),
        "mnist": partial(TfdsWrapper, name="mnist"),
        "pcam": partial(TfdsWrapper, name="patch_camelyon"),
        "clevr": CLEVRCounts,
        # "country211": partial(Country211, download=True),
        # "sst2": partial(RenderedSST2, download=True),
        "imagenet": ImageNet,
        "imagenet_a": partial(TfdsWrapper, name="imagenet_a"),
        "imagenet_r": partial(TfdsWrapper, name="imagenet_r"),
        "imagenet_v2": partial(TfdsWrapper, name="imagenet_v2"),
        "imagenet_sketch": partial(TfdsWrapper, name="imagenet_sketch"),
    }

    # List of names of [train, val, test] splits:
    SPLITS: dict[str, list[str]] = {
        "food101": ["train[:80%]", "train[80%:]", "validation"],
        "cifar10": ["train[:80%]", "train[80%:]", "test"],
        "cifar100": ["train[:80%]", "train[80%:]", "test"],
        "cub2011": ["train[:80%]", "train[80%:]", "test"],
        "sun397": ["train[:80%]", "train[80%:]", "test"],
        "cars196": ["train[:80%]", "train[80%:]", "test"],
        "aircraft": ["train", "val", "test"],
        "dtd": ["train", "validation", "test"],
        "pets": ["train[:80%]", "train[80%:]", "test"],
        "caltech101": ["train[:80%]", "train[80%:]", "test"],
        "flowers": ["train", "validation", "test"],
        "stl10": ["train[:80%]", "train[80%:]", "test"],
        "eurosat": ["train[:5000]", "train[5000:10000]", "train[10000:15000]"],
        "resisc45": ["train[:10%]", "train[10%:20%]", "train[20%:]"],
        "mnist": ["train[:80%]", "train[80%:]", "test"],
        "pcam": ["train", "validation", "test"],
        "clevr": ["train[:1500]", "train[1500:2000]", "validation[:500]"],
        # "country211": ["train", "valid", "test"],
        # "sst2": ["train", "val", "test"],
        "imagenet": ["train", "", "val"],
        "imagenet_a": ["", "", "test"],
        "imagenet_r": ["", "", "test"],
        "imagenet_v2": ["", "", "test"],
        "imagenet_sketch": ["", "", "test"],
    }

    METRICS: dict[str, Callable] = {
        "food101": partial(Accuracy, num_classes=101),
        "cifar10": partial(Accuracy, num_classes=10),
        "cifar100": partial(Accuracy, num_classes=100),
        "cub2011": partial(Accuracy, num_classes=200),
        "sun397": partial(Accuracy, num_classes=397),
        "cars196": partial(Accuracy, num_classes=196),
        "aircraft": partial(Accuracy, num_classes=100, mean_per_class=True),
        "dtd": partial(Accuracy, num_classes=47),
        "pets": partial(Accuracy, num_classes=37, mean_per_class=True),
        "caltech101": partial(Accuracy, num_classes=102, mean_per_class=True),
        "flowers": partial(Accuracy, num_classes=102, mean_per_class=True),
        "stl10": partial(Accuracy, num_classes=10),
        "eurosat": partial(Accuracy, num_classes=10),
        "resisc45": partial(Accuracy, num_classes=45),
        "mnist": partial(Accuracy, num_classes=10),
        "pcam": partial(Accuracy, num_classes=2),
        "clevr": partial(Accuracy, num_classes=8),
        # "country211": partial(Accuracy, num_classes=211),
        # "sst2": partial(Accuracy, num_classes=2),
        "imagenet": partial(Accuracy, num_classes=1000),
    }

    @classmethod
    def build_dataset(
        cls, name: str, root: str | Path, split: str, transform: Callable | None = None
    ):
        if name not in cls.CONSTRUCTORS:
            supported = sorted(cls.CONSTRUCTORS.keys())
            raise ValueError(f"{name} is not among supported datasets: {supported}")

        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be one of [train, val, test], not {split}")

        # Change the root directory for some Torchvision datasets because their
        # auto-download location may clutter the dataset directory.
        if name in ["aircraft", "country211", "imagenet", "sst2"]:
            root = str(Path(root) / name)

        # Map split from [train, val, test] to official name.
        _idx = ["train", "val", "test"].index(split)
        split = cls.SPLITS[name][_idx]

        return cls.CONSTRUCTORS[name](root=root, split=split, transform=transform)

    @classmethod
    def build_metric(cls, name: str):
        if name not in cls.METRICS:
            supported = sorted(cls.METRICS.keys())
            raise ValueError(f"{name} is not among supported datasets: {supported}")

        return cls.METRICS[name]()
