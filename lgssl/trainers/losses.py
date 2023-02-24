import torch
import torch.nn as nn
import torch.nn.functional as F

from lgssl.utils.distributed import all_gather_batch_with_grad, get_rank


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.local_bs = None
        self.start_i = None

    def forward(self, img_embed, txt_embed, logit_scale):
        img_embed = F.normalize(img_embed, dim=-1, p=2)
        txt_embed = F.normalize(txt_embed, dim=-1, p=2)

        local_bs = img_embed.size(0)

        # gather features from all GPUs
        tensors_all, start_i = all_gather_batch_with_grad([img_embed, txt_embed])
        img_embed_all, txt_embed_all = tensors_all

        if local_bs != self.local_bs or start_i != self.start_i:
            self.labels = start_i + torch.arange(local_bs, device=img_embed.device)
            self.local_bs = local_bs
            self.start_i = start_i

        # cosine similarity as logits -- scale is multiplicative
        logits_per_img = img_embed @ txt_embed_all.t() * logit_scale
        logits_per_txt = txt_embed @ img_embed_all.t() * logit_scale

        loss_v = F.cross_entropy(logits_per_img, self.labels)
        loss_l = F.cross_entropy(logits_per_txt, self.labels)

        loss = (loss_v + loss_l) / 2.0

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_img, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_bs

        return loss, acc


class SimCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.local_bs = None
        self.total_bs = None
        self.start_i = None

    def forward(self, view_a, view_b):
        view_a = F.normalize(view_a, dim=-1, p=2)
        view_b = F.normalize(view_b, dim=-1, p=2)

        local_bs = view_a.size(0)

        k_all, start_i = all_gather_batch_with_grad([view_a, view_b])
        k_a, k_b = k_all
        total_bs = k_a.shape[0]

        matching_bs = local_bs != self.local_bs or self.total_bs != total_bs
        if matching_bs or start_i != self.start_i:
            self.labels = start_i + torch.arange(local_bs, device=view_a.device)
            self.masks = F.one_hot(self.labels, total_bs) * 1e9

            self.local_bs = local_bs
            self.start_i = start_i
            self.total_bs = total_bs

        # compute pairwise logits
        logits_aa = torch.matmul(view_a, k_a.transpose(0, 1)) / self.tau
        logits_bb = torch.matmul(view_b, k_b.transpose(0, 1)) / self.tau
        logits_ab = torch.matmul(view_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(view_b, k_a.transpose(0, 1)) / self.tau

        # remove self from logits by subtracting out a very big number
        logits_aa = logits_aa - self.masks
        logits_bb = logits_bb - self.masks

        # Form full logit matrix for cross_entropy
        full_logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        full_logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        # compute cross entropy with 2N - 1 negatives
        loss_a = F.cross_entropy(full_logits_a, self.labels)
        loss_b = F.cross_entropy(full_logits_b, self.labels)

        loss = (loss_a + loss_b) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_bs

        return loss, acc


class NNCLRLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.local_bs = None
        self.start_i = None

    def forward(self, proj_a, proj_b, pred_a, pred_b):
        proj_a = F.normalize(proj_a, dim=-1, p=2)
        proj_b = F.normalize(proj_b, dim=-1, p=2)
        pred_a = F.normalize(pred_a, dim=-1, p=2)
        pred_b = F.normalize(pred_b, dim=-1, p=2)

        local_bs = pred_a.size(0)

        proj_all, start_i = all_gather_batch_with_grad([proj_a, proj_b])
        proj_a, proj_b = proj_all

        if local_bs != self.local_bs or start_i != self.start_i:
            self.labels = start_i + torch.arange(local_bs, device=proj_a.device)
            self.local_bs = local_bs
            self.start_i = start_i

        if local_bs != self.local_bs:
            self.labels = local_bs * get_rank() + torch.arange(
                local_bs, device=pred_a.device
            )
            self.local_bs = local_bs

        # compute pairwise logits
        logits_ab = torch.matmul(pred_a, proj_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(pred_b, proj_a.transpose(0, 1)) / self.tau

        # compute cross entropy with 2N - 1 negatives
        loss_a = F.cross_entropy(logits_ab, self.labels)
        loss_b = F.cross_entropy(logits_ba, self.labels)

        loss = (loss_a + loss_b) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_ab, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_bs

        return loss, acc
