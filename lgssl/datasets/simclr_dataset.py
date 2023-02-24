import json
from pathlib import Path

from tqdm import tqdm

from .base_dataset import BaseDataset


class SimCLRDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        sampling: str,
        augmentation: str,
        nn_encoder: str = "all-mpnet-base-v2",
        similarity_threshold: float = 0.0,
        ratio_threshold: float = 0.0,
        return_captions: bool = True,
        **kwargs,  # Allow arbitrary kwargs without raising errors.
    ):
        super().__init__(name)

        # get instances
        self.sampling = sampling
        self.nn_encoder = nn_encoder
        self.similarity_threshold = similarity_threshold
        self.ratio_threshold = ratio_threshold
        self.return_captions = return_captions

        if sampling == "self":
            self.instances = self.get_dict_instances()

            # Print out dataset stats
            print(f"Self-sample dataset: {self.name}")
            print(f"Numer of instances {len(self.instances)}")

        elif sampling == "nn":
            self.instances = self.get_nn_instances()

            # Print out dataset stats
            print(f"Nearest-neighbor sampled: {self.name}")
            print(f"Sampling in {nn_encoder}:")
            print(f"    similarity threshold:   {similarity_threshold}")
            print(f"    ratio threshold:        {ratio_threshold}")
            print(f"Numer of instances {len(self.instances)}")

        self.cpu_augment, self.gpu_augment = self.get_augmentation(augmentation)

    def __getitem__(self, index):
        if self.sampling == "self":
            return self.get_self_sample(index)
        elif self.sampling == "nn":
            return self.get_nn_sample(index)
        else:
            raise ValueError()

    def get_self_sample(self, index):
        tar_id, img_id, path, caption = self.instances[index]

        output = {
            "uid": f"{tar_id}-{img_id}",
            "path_0": path,
            "path_1": path,
            "augmentation": self.gpu_augment,
        }

        if self.return_captions:
            output["caption_0"] = caption
            output["caption_1"] = caption

        input_image = self.load_image(path)

        try:
            output["image_0"] = self.cpu_augment(input_image)
            output["image_1"] = self.cpu_augment(input_image)
        except:
            return None

        return output

    def get_nn_sample(self, index):
        inst_0, inst_1, nn_sim = self.instances[index]

        if self.return_captions:
            tar_id_0, img_id_0, path_0, caption_0 = inst_0
            tar_id_1, img_id_1, path_1, caption_1 = inst_1
        else:
            tar_id_0, img_id_0, path_0 = inst_0
            tar_id_1, img_id_1, path_1 = inst_1

        output = {
            "uid": f"{tar_id_0}-{img_id_0}_{tar_id_1}-{img_id_1}",
            "path_0": path_0,
            "path_1": path_1,
            "augmentation": self.gpu_augment,
            "nn_similarity": nn_sim,
        }

        if self.return_captions:
            output["caption_0"] = caption_0
            output["caption_1"] = caption_1

        input_image_0 = self.load_image(path_0)
        input_image_1 = self.load_image(path_1)

        try:
            output["image_0"] = self.cpu_augment(input_image_0)
            output["image_1"] = self.cpu_augment(input_image_1)
        except:
            return None

        return output

    def get_nn_instances(self):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # load data file and nn file
        DICT_ROOT = Path(__file__).parent.parent / "../data/data_dicts"
        data_file = DICT_ROOT / f"{self.name}.json"
        lsnn_file = DICT_ROOT / f"{self.name}_{self.nn_encoder}.json"
        print("The file is: ", lsnn_file)

        data_dict = json.load(data_file.open())
        lsnn_list = json.load(lsnn_file.open())

        # get dictionary
        for inst in tqdm(lsnn_list, desc="Extract instances"):
            if len(inst) == 3:
                source, target, nn_sim = inst
                nn_ratio = 1.0
            else:
                source, target, nn_sim, nn_ratio = inst

            s_tar, s_img_id = source[:2]
            t_tar, t_img_id = target[:2]

            if nn_sim < self.similarity_threshold or nn_ratio < self.ratio_threshold:
                continue

            s_path, s_caption = data_dict[s_tar][s_img_id][0:2]
            t_path, t_caption = data_dict[t_tar][t_img_id][0:2]

            if self.return_captions:
                s_ins = (s_tar, s_img_id, s_path, s_caption)
                t_ins = (t_tar, t_img_id, t_path, t_caption)
            else:
                s_ins = (s_tar, s_img_id, s_path)
                t_ins = (t_tar, t_img_id, t_path)

            instances.append((s_ins, t_ins, nn_sim))

        return instances
