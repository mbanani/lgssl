dependencies = ["torch", "timm"]

import torch
import timm


# fmt: off
CKPT_URLS = {
    # Get file name as URL key:
    url.split("/")[-1].replace(".ckpt", ""): url
    for url in [
        "https://www.dropbox.com/sh/me6nyiewlux1yh8/AADWGD0bprAVVQdMkeQTe-N8a/clip_baseline_nns.ckpt",
        "https://www.dropbox.com/sh/me6nyiewlux1yh8/AABfJIHNWmXWKiqEgvtRcgtua/clip_baseline_sbert_as_text_enc.ckpt",
        "https://www.dropbox.com/sh/me6nyiewlux1yh8/AADcT7i8tO9vckWb0AX1v9rYa/clip_baseline.ckpt",
        "https://www.dropbox.com/sh/me6nyiewlux1yh8/AAC63z4OHBvfrKSPRItZHK0Ca/lg_simclr_bs_256.ckpt",
        "https://www.dropbox.com/sh/me6nyiewlux1yh8/AADvIWjm07m31JxoPnGeazZva/lg_simclr_bs_1024.ckpt",
        "https://www.dropbox.com/sh/me6nyiewlux1yh8/AAC3YuEdAnsmKauI2XSfJMKUa/lg_simclr_bs_2048.ckpt",
    ]
}
# fmt: on


# Programatically generate a lot of functions in global scope.
for name, url in CKPT_URLS:
    exec(
        f"""
def {name}():
    model = timm.create_model("resnet50", num_classes=0)
    model.load_state_dict(torch.hub.load_state_dict_from_url({url}))
    return model
        """
    )
