import argparse
import json
import os
import time
from pathlib import Path

import clip
import kornia.augmentation as k_transforms
import torch
import torchvision.transforms as tv_transforms
from omegaconf import OmegaConf
from sample_language_nn import sample_pairs
from torch.nn.functional import normalize
from tqdm import tqdm

from lgssl.datasets.builder import build_loader


def extract_vision_embeddings(dataset_cfg, model_name, model_weights):
    print("---- get dataset ----")
    curr_time = time.time()
    dataset_name = dataset_cfg.name
    n_workers = len(os.sched_getaffinity(0))
    loader = build_loader(dataset_cfg, shuffle=False, n_workers=n_workers)
    print(f"Loaded dataset {dataset_name} in {time.time() - curr_time:.2f}sec")

    print("---- generate embeddings ----")
    curr_time = time.time()
    if model_name == "clip":
        clip_model, _ = clip.load(model_weights, "cuda")
        model = clip_model.encode_image
        img_mean = [0.48145466, 0.4578275, 0.40821073]
        img_std = [0.26862954, 0.26130258, 0.27577711]
    else:
        model = torch.hub.load("pytorch/vision", model_name, weights=model_weights)
        model.fc = torch.nn.Identity()
        model.heads = torch.nn.Identity()
        model = model.cuda().eval()
        model = torch.nn.DataParallel(model)
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]

    dataset = []
    embeddings = []
    gpu_augment = torch.nn.Sequential(
        tv_transforms.ConvertImageDtype(torch.float32),
        k_transforms.Normalize(mean=img_mean, std=img_std),
    )

    # set embedding size for now
    dataset_size = len(loader.dataset)
    embedding_size = None
    prev_end = 0

    for i, batch in enumerate(tqdm(loader)):
        uids = batch["uid"]
        imgs = batch["image_0"]
        imgs = gpu_augment(imgs)
        caps = batch["caption_0"]
        batch_size = imgs.shape[0]

        for _j in range(batch_size):
            tar_i, id_i = uids[_j].split("-")
            dataset.append((tar_i, id_i, caps[_j]))

        with torch.inference_mode():
            emb_i = model(imgs.cuda())
            if type(emb_i) is list:
                emb_i = emb_i[0]

            if embedding_size is None:
                assert len(emb_i.shape) == 2
                embedding_size = emb_i.shape[1]
                embeddings = torch.zeros(
                    dataset_size, embedding_size, dtype=torch.float
                )
            else:
                assert embedding_size == emb_i.shape[1], emb_i.shape

            # get indices
            start_i = prev_end
            end_i = prev_end + batch_size
            prev_end = end_i
            assert end_i <= dataset_size
            embeddings[start_i:end_i] = normalize(emb_i, p=2, dim=-1).detach().cpu()
            del emb_i

    del model
    torch.cuda.empty_cache()

    embeddings = embeddings[:prev_end]
    assert len(dataset) == prev_end
    print(f"Gathered embeddings in {time.time() - curr_time:.2f}sec")

    return dataset, embeddings


def generate_dict(dataset_cfg, model_name, model_weights):

    # extract embeddings
    dataset, embeddings = extract_vision_embeddings(
        dataset_cfg, model_name, model_weights
    )

    # sample pairs
    nn_pairs = sample_pairs(dataset, embeddings)

    # change name
    model_weights = model_weights.replace("/", "").replace("-", "").lower()
    encoder = f"vis_{model_name}-{model_weights}"

    # save dict
    dataset_name = dataset_cfg.name
    save_dir = Path(__file__).parent / "../data/data_dicts"
    save_path = save_dir / f"{dataset_name}_{encoder}.json"

    json.dump(nn_pairs, save_path.open("w"))
    print(f"Saved nearest neighbors to: {str(save_path.resolve())}")


if __name__ == "__main__":
    # --- define data ---

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("weights", type=str)
    args = parser.parse_args()

    # define dataset config
    dataset_cfg = OmegaConf.create(
        {
            "_target_": "lgssl.datasets.simclr_dataset.SimCLRDataset",
            "name": args.dataset,
            "sampling": "self",
            "augmentation": "center_crop",
            "batch_size": 2048,
        }
    )

    generate_dict(dataset_cfg, args.model, args.weights)
