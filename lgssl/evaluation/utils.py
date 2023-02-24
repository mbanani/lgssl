from pathlib import Path

import clip
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm


def clean_state_dict(state_dict, remove_prefix):
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith(remove_prefix):
            state_dict[k[len(remove_prefix) :]] = state_dict[k]
        del state_dict[k]
    return state_dict


def get_model(model_cfg):  # noqa: C901
    model_name = model_cfg.name
    ckpt_name = model_cfg.checkpoint
    if ckpt_name in ["random", "IMAGENET1K_V1", "IMAGENET1K_V2"]:
        ckpt_name = None if ckpt_name == "random" else ckpt_name
        model = torchvision.models.__dict__[model_name](weights=ckpt_name)
        model.fc = torch.nn.Identity()
    elif model_name in ["simsiam", "simclr", "mocov3", "swav"]:
        ckpt_info = {
            "simsiam": ("simsiam.pth", "module.encoder."),
            "simclr": ("simclr.ckpt", "encoder."),
            "swav": ("swav.pth", "module."),
            "mocov3": ("mocov3.pth", "module.base_encoder."),
        }

        ckpt_file, prefix = ckpt_info[model_name]
        ckpt_path = Path(__file__).parent / "baseline_weights" / ckpt_file
        ckpt_path = str(ckpt_path.resolve())
        ckpt = torch.load(ckpt_path, map_location="cpu")

        state_dict = ckpt if model_name == "swav" else ckpt["state_dict"]
        state_dict = clean_state_dict(state_dict, prefix)

        model = torchvision.models.__dict__["resnet50"]()
        msg = model.load_state_dict(state_dict, strict=False)
        if model_name != "simclr":
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        else:
            assert len(msg.missing_keys) == 0

        model.fc = torch.nn.Identity()
    elif model_name == "clip":
        assert model_name == "resnet50"
        model, _ = clip.load("RN50", device="cpu")
        model = model.visual
    elif model_name == "lgssl_checkpoints":
        ckpt_path = Path(__file__) / f"../../../data/checkpoints/{ckpt_name}.ckpt"
        ckpt_path = str(ckpt_path.resolve())

        state_dict = torch.load(ckpt_path, map_location="cpu")
        model = torchvision.models.__dict__["resnet50"]()
        msg = model.load_state_dict(state_dict, strict=False)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError()

    model = model.eval().cuda()
    return model


def extract_features(
    model, loader, normalize=False, norm_stats=None, return_stats=False
):
    """
    Extract global average pooled visual features for linear probe evaluation.
    Args:
        model: Trained model with ``visual`` module for feature extraction.
        dataset laoder: Dataset loader to serve ``(image, label)`` tuples.
    """
    feature_label_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for images, labels in tqdm(loader, desc="Extracting feats"):
        with torch.inference_mode():
            images = images.cuda()
            features = model(images)
            if type(features) is list:
                assert len(features) == 1
                features = features[0]
            feature_label_pairs.append((features.cpu(), labels))

    all_features = torch.cat([p[0] for p in feature_label_pairs], dim=0)
    all_labels = torch.cat([p[1] for p in feature_label_pairs], dim=0)

    if normalize:
        if norm_stats is None:
            feature_mean = all_features.mean(dim=0, keepdim=True)
            feature_std = all_features.std(dim=0, keepdim=True)
        else:
            feature_mean, feature_std = norm_stats

        all_features = (all_features - feature_mean) / feature_std

    if return_stats:
        return all_features, all_labels, (feature_mean, feature_std)
    else:
        return all_features, all_labels
