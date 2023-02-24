import json
from pathlib import Path

from .base_dataset import BaseDataset


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        augmentation: str,
        encoder: str = "all-MiniLM-L12-v2",
        n_class: int = 1000,
        **kwargs,  # Allow arbitrary kwargs without raising errors.
    ):
        super().__init__(name)
        # "cc3m_all-MiniLM-L12-v2_1000classes_train.json"
        # get instances
        self.encoder = encoder
        self.n_class = n_class
        self.instances = self.get_labeled_dict_instances()

        # Print out dataset stats
        print(f"Classification (Image-Label) Dataset: {self.name}")
        print(f"Numer of instances {len(self.instances)}")

        # define two augmentations; one run on CPU loader and one batched on GPU
        self.cpu_augment, self.gpu_augment = self.get_augmentation(augmentation)

    def __getitem__(self, index):
        tar_id, img_id, path, caption, label = self.instances[index]

        # process first image
        input_image = self.load_image(path)

        try:
            image = self.cpu_augment(input_image)
        except:
            return None

        uid = f"{tar_id}-{img_id}"

        return {
            "uid": uid,
            "image_0": image,
            "path_0": path,
            "caption_0": caption,
            "label_0": label,
            "augmentation": self.gpu_augment,
        }

    def get_labeled_dict_instances(self):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # load data file and nn file
        DICT_ROOT = Path(__file__).parent.parent / "data/data_dicts"
        data_file = f"{self.name}.json"
        cls_file = f"{self.name}_{self.encoder}_{self.n_class}classes.json"

        data_dict = json.load(data_file.open())
        class_list = json.load(cls_file.open())

        for inst in class_list:
            tar, img_id = inst[0]
            label = inst[1]

            path, caption = data_dict[tar][img_id][0:2]

            instances.append((tar, img_id, path, caption, label))

        return instances
