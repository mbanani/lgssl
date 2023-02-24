# Code based on sampler from @mileyan/simple_shot
import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import Sampler
from tqdm import tqdm


def evaluate_fewshot(feats_train, labels_train, feats_test, labels_test, fs_cfg):
    # set up sampler
    fewshot_sampler = FewShotEpisodeSampler(
        labels_train,
        labels_test,
        fs_cfg.n_iter,
        fs_cfg.n_way,
        fs_cfg.n_shot,
        fs_cfg.n_query,
    )

    # test model on dataset -- really more tasks than batches
    accuracy = []
    n_way = fs_cfg.n_way
    n_shot = fs_cfg.n_shot

    for task in tqdm(fewshot_sampler):
        source, query = task

        # get train and test
        feats_source = feats_train[source]
        labels_source = labels_train[source]
        feats_query = feats_test[query]
        labels_query = labels_test[query]

        # center
        if fs_cfg.center_feats:
            feats_mean = feats_source.mean(dim=0, keepdims=True)
            feats_query = feats_query - feats_mean
            feats_source = feats_source - feats_mean

        # normalize
        if fs_cfg.normalize_feats:
            feats_source = normalize(feats_source, dim=-1, p=2)
            feats_query = normalize(feats_query, dim=-1, p=2)

        # compute prototypes & assert labels are correct
        if fs_cfg.average_feats:
            feats_proto = feats_source.view(n_way, n_shot, -1).mean(dim=1)
            labels_proto = labels_source.view(n_way, n_shot)
            try:
                assert (
                    labels_proto.min(dim=1).values == labels_proto.max(dim=1).values
                ).all()
            except:
                breakpoint()
            labels_proto = labels_proto[:, 0]
        else:
            feats_proto = feats_source
            labels_proto = labels_source

        # classify to prototypes
        pw_dist = (feats_query[:, None] - feats_proto[None, :]).norm(dim=-1, p=2)

        labels_pred = labels_proto[pw_dist.min(dim=1).indices]
        is_correct = (labels_pred == labels_query).float().mean()

        accuracy.append(is_correct)

    # compute metrics for model
    accuracy = torch.stack(accuracy) * 100.0
    return accuracy.mean(), accuracy.std()


class FewShotEpisodeSampler(Sampler):
    def __init__(self, train_labels, test_labels, n_iter, n_way, n_shot, n_query):
        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        train_labels = np.array(train_labels)
        self.train_ind = []
        self.test_ind = []
        unique = np.unique(train_labels)
        unique = np.sort(unique)
        for i in unique:
            train_ind = np.argwhere(train_labels == i).reshape(-1)
            self.train_ind.append(train_ind)

            test_ind = np.argwhere(test_labels == i).reshape(-1)
            self.test_ind.append(test_ind)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.train_ind))[: self.n_way]
            for c in classes:
                train_c = self.train_ind[c.item()]
                assert len(train_c) >= (self.n_shot), f"{len(train_c)} < {self.n_shot}"
                train_pos = torch.multinomial(torch.ones(len(train_c)), self.n_shot)
                batch_gallery.append(train_c[train_pos])

                test_c = self.test_ind[c.item()]
                if len(test_c) < (self.n_query):
                    print(f"test class has {len(test_c)} ins. (< {self.n_query})")
                    batch_query.append(test_c)
                else:
                    test_pos = torch.multinomial(torch.ones(len(test_c)), self.n_query)
                    batch_query.append(test_c[test_pos])

            batch_gallery = np.concatenate(batch_gallery)
            batch_query = np.concatenate(batch_query)

            yield (batch_gallery, batch_query)
