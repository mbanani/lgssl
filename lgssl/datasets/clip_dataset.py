from .base_dataset import BaseDataset


class CLIPDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        augmentation: str,
        **kwargs,  # Allow arbitrary kwargs without raising errors.
    ):
        super().__init__(name)

        # get instances
        self.instances = self.get_dict_instances()

        # Print out dataset stats
        print(f"CLIP (Image-Caption) Dataset: {self.name}")
        print(f"Numer of instances {len(self.instances)}")

        # define two augmentations; one run on CPU loader and one batched on GPU
        self.cpu_augment, self.gpu_augment = self.get_augmentation(augmentation)

    def __getitem__(self, index):
        tar_id, img_id, path, caption = self.instances[index]

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
            "augmentation": self.gpu_augment,
        }
