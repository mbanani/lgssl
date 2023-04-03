import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from .sample_language_nn import get_dataset, subsample_dataset


def generate_dict(dataset_name, encoder, subset):
    print("---- get dataset ----")
    curr_time = time.time()
    dataset = get_dataset(f"{dataset_name}.json")
    print(f"Gathered {len(dataset)} captions in {time.time() - curr_time:.2f}sec")

    # find categories to split
    if subset == "subreddit":
        assert "redcaps" in dataset_name
        dir_names = [x[0].split("_")[0] for x in dataset]
    elif subset == "year":
        assert "redcaps" in dataset_name
        dir_names = [x[0].split("_")[1] for x in dataset]
    if subset == "subreddityear":
        assert "redcaps" in dataset_name
        dir_names = ["_".join(x[0].split("_")[:2]) for x in dataset]
    else:
        raise ValueError()

    unique = set(dir_names)
    dataset_splits = {}
    for i in tqdm(range(len(dataset))):
        dir_i = dir_names[i]
        if dir_i not in dataset_splits:
            dataset_splits[dir_i] = []

        dataset_splits[dir_i].append(dataset[i])

    assert len(unique) == len(dataset_splits)
    for subset_name in dataset_splits:
        print(f"split {subset_name} has {len(dataset_splits[subset_name])} instances")

    # extract pairs
    data_subsets = None
    subset_names = list(dataset_splits.keys())
    num_subsets = len(subset_names)
    for i in tqdm(range(num_subsets)):
        subset_k = subset_names[i]
        nn_subset = subsample_dataset(dataset_splits[subset_k], encoder, 1, False)
        if data_subsets is None:
            data_subsets = nn_subset
        else:
            data_subsets += nn_subset

    # save dict
    save_dir = Path(__file__).parent / "../data/data_dicts"
    save_path = save_dir / f"{dataset_name}_{encoder}-{subset}.json"
    json.dump(data_subsets, save_path.open("w"))


if __name__ == "__main__":
    # --- define data ---

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("encoder", type=str)
    parser.add_argument("subset", type=str)
    args = parser.parse_args()

    generate_dict(args.dataset, args.encoder, args.subset)
