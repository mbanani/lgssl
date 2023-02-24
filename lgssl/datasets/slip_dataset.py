from .base_dataset import BaseDataset


class SLIPDataset(BaseDataset):
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
        print(f"SLIP (Image-Caption) Dataset: {self.name}")
        print(f"Numer of instances {len(self.instances)}")

        # define two augmentations; one run on CPU loader and one batched on GPU
        assert augmentation == "slip"
        self.v_cpu_augment, self.v_gpu_augment = self.get_augmentation("simclr")
        self.vl_cpu_augment, self.vl_gpu_augment = self.get_augmentation("global_crop")

    def __getitem__(self, index):
        tar_id, img_id, path, caption = self.instances[index]

        # process first image
        input_image = self.load_image(path)

        try:
            image_t0 = self.v_cpu_augment(input_image)
            image_t1 = self.v_cpu_augment(input_image)
            image_tc = self.vl_cpu_augment(input_image)
        except:
            return None

        uid = f"{tar_id}-{img_id}"

        return {
            "uid": uid,
            "image_t0": image_t0,
            "image_t1": image_t1,
            "image_tc": image_tc,
            "path_0": path,
            "caption_0": caption,
            "simclr_augmentation": self.v_gpu_augment,
            "clip_augmentation": self.vl_gpu_augment,
        }
