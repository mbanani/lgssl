import argparse
import glob
import json
import os
from pathlib import Path

from tqdm import tqdm


def get_dataset(dataset_name, split, year=None):
    datasets_root = Path(__file__).parent / "../data/datasets"
    dataset_path = (datasets_root / dataset_name / split).resolve()
    shards = os.listdir(dataset_path)

    def check_year(shard):
        tname = shard.split("/")[-1].split(".")[0]
        return str(year) in tname

    if year is not None:
        shards = [shard for shard in shards if check_year(shard)]

    dataset = {}
    shards.sort()
    for shard in tqdm(shards):
        shard_path = str((dataset_path / shard).resolve())
        annot_files = glob.glob(f"{shard_path}/*.json")
        shard_imgs = os.listdir(shard_path)
        shard_imgs = [_file for _file in shard_imgs if "json" not in _file]
        shard_imgs = {_img.split(".")[0]: _img for _img in shard_imgs}

        # parse sub-directory for paths and ids
        shard_dict = {}
        skipped = 0
        for cap_path in annot_files:
            inst_id = cap_path.split("/")[-1].split(".")[0]

            if inst_id in shard_imgs:
                with open(cap_path) as f:
                    caption = json.load(f)["caption"]

                # image relative path to dataset root
                img_path = f"{shard}/{shard_imgs[inst_id]}"
                shard_dict[inst_id] = (img_path, caption)
            else:
                skipped += 1

        if skipped > 0:
            print(f"{shard}: {skipped} skipped.")

        # add shard dict to full dataset dict
        if len(shard_dict) > 0:
            dataset[shard] = shard_dict

    # save dict
    data_dicts_path = Path(__file__).parent / "../data/data_dicts"
    if year is not None:
        save_path = data_dicts_path / f"{dataset_name}-{year}.json"
    else:
        save_path = data_dicts_path / f"{dataset_name}.json"

    json.dump(dataset, save_path.open("w"))
    return dataset


if __name__ == "__main__":
    # --- define data ---
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()

    get_dataset(args.dataset, args.split, args.year)
