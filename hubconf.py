dependencies = ["torch", "timm"]

import inspect
import torch
import timm


# fmt: off
# ------------------------------------------------------------------------------------------------
# Our main language-guided SSL models:
def lgsimclr():                        return _load_model("AADlHQyayNwjfSiYqumOf3VAa")
def lgsimsiam():                       return _load_model("AAAETt_6yjLLO9_vqiLOPw1-a")
def lgslip():                          return _load_model("AACDoCE9xfzbi8nQwffd0pe7a")

# Our LG-SimCLR model with variable batch sizes.
def lgsimclr_bs_256():                 return _load_model("AADjYsCNy7mT0J_XboDmjUmAa")
def lgsimclr_bs_1024():                return _load_model("AADjRzzptn8g-oJUcdTi5CUca")
def lgsimclr_bs_2048():                return _load_model("AACjJMgHMRqOEPw-lQ2QtSMka")
lgsimclr_bs_512 = lgsimclr

# ------------------------------------------------------------------------------------------------
# Visual baselines: ones that use image-image contrastive learning with augmentations.
def visual_baseline_simsiam():         return _load_model("AAAZPvJNoulGFwujA4gUroyNa")
def visual_baseline_nnclr():           return _load_model("AACnmKI8H-jBDVfmIn02183-a")
def visual_baseline_simclr():          return _load_model("AADke3kbdVXq5On42j6t9PWqa")

# CLIP baselines: ones that use image-text contrastive learning.
def clip_baseline():                   return _load_model("AADcT7i8tO9vckWb0AX1v9rYa")
def clip_baseline_nns():               return _load_model("AADWGD0bprAVVQdMkeQTe-N8a")
def clip_baseline_sbert_as_text_enc(): return _load_model("AABfJIHNWmXWKiqEgvtRcgtua")

# SLIP baseline: combines image-image and image-text contrastive learning objectives.
def slip_baseline():                   return _load_model("AACKJlso6EfHH5nS9bbsrEZha")

# SimCLR baseline with variable batch sizes.
def simclr_bs_256():                   return _load_model("AAB9PEHrzJPYSi0HfsmqHKIpa")
def simclr_bs_1024():                  return _load_model("AAD20wcCgXkFN-QKInsbu-iIa")
def simclr_bs_2048():                  return _load_model("AAC6Mind8XO9GwFK2-2QKZvpa")

# ------------------------------------------------------------------------------------------------
# Extra LG-SimCLR models: trained with different sampling spaces in RedCaos (SBERT encoder).
def redcaps_subsample_subreddit():     return _load_model("AAD32C8YahqHUXtvbFLZnGHea")
def redcaps_subsample_subreddityear(): return _load_model("AABb5Z1q10qfy_PHK-sH_DhWa")
def redcaps_subsample_year():          return _load_model("AADHjc_6JkimJ0ua5dL4RB3Ya")

# Extra LG-SimCLR models: with nearest neighbors sampled using pre-trained visual encoders.
def sample_space_vis_clip():           return _load_model("AAArta3X3DfW7Lr-1yQh5-o7a")
def sample_space_vis_imagenet():       return _load_model("AACf5f0TRnkpwddzizFxrp_Ya")
def sample_space_vis_simclr():         return _load_model("AADLvWEQfCb5QcuNSNeemM5Pa")

# Extra LG-SimCLR models: with nearest neighbors sampled using pre-trained language encoders.
def sample_space_lang_clip_vitb32():   return _load_model("AABTtKYcCyGAe_gx7HGiITK_a")
def sample_space_lang_fasttext_bow():  return _load_model("AABlIN293fJ9RYpduHwbtnzua")
def sample_space_lang_minilm():        return _load_model("AAAe7NxKJ7Ut_ESFA1u7C-bMa")
def sample_space_lang_our_clip():      return _load_model("AABom01m4OMqb0LmP5mBlNfFa")

# ------------------------------------------------------------------------------------------------
# fmt: on


def _load_model(dropbox_id: str):
    base_url = "https://www.dropbox.com/sh/me6nyiewlux1yh8"

    # Dark magic to get name of the function that called this function.
    # This line determines the name of model in model zoo.
    model_name = inspect.getouterframes(inspect.currentframe())[1][3]

    model = timm.create_model("resnet50", num_classes=0)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            f"{base_url}/{dropbox_id}/{model_name}.ckpt?dl=1"
        )
    )
    return model

