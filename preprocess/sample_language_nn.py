import argparse
import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize
from tqdm import tqdm

from lgssl.utils.faiss import faiss_knn


def truncate_caption(sentence, tokenizer):
    """
    Truncate a sentence to fit the CLIP max token limit (77 tokens including the
    starting and ending tokens).

    Args:
        sentence(string): The sentence to truncate.
        tokenizer(CLIPTokenizer): Rretrained CLIP tokenizer.
    """

    cur_sentence = sentence
    tokens = tokenizer.encode(cur_sentence)

    if len(tokens) > 77:
        # Skip the starting token, only include 75 tokens
        truncated_tokens = tokens[1:76]
        cur_sentence = tokenizer.decode(truncated_tokens)

        # Recursive call here, because the encode(decode()) can have different result
        return truncate_caption(cur_sentence, tokenizer)

    else:
        return cur_sentence


def embed_captions(captions, model_name, batch_size, num_gpus=1):
    model = SentenceTransformer(model_name, device="cuda")

    # there's a weird bug with long captions and SBERT's CLIP
    if "clip" in model_name:
        model.max_seq_length = 77
        tokenizer = model._first_module().processor.tokenizer
        print("Truncate captions")
        captions = [truncate_caption(_c, tokenizer) for _c in tqdm(captions)]

    print(f"Loaded model {model_name} with max_length: {model.max_seq_length}")
    print("Embedding captions ...")

    with torch.inference_mode():
        if num_gpus > 1:
            # start the multi-process pool on all available CUDA devices
            pool = model.start_multi_process_pool()

            # Compute the embeddings using the multi-process pool
            inc = 100000
            embs = []
            for i in tqdm(range(0, len(captions), inc), ncols=80):
                end = min(i + inc, len(captions))
                cap_i = captions[i:end]
                emb_i = model.encode_multi_process(cap_i, pool, batch_size=batch_size)
                embs.append(torch.tensor(emb_i))

            # Optional: Stop the proccesses in the pool
            embeddings = torch.cat(embs, dim=0)
            model.stop_multi_process_pool(pool)
        else:
            embeddings = model.encode(
                captions,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=False,
            )
            embeddings = torch.stack(embeddings, dim=0).cpu()

    del model
    torch.cuda.empty_cache()

    return embeddings


def extract_language_embeddings(dataset_name, encoder):
    print("---- get dataset ----")
    curr_time = time.time()
    dict_path = Path(__file__).parent / f"../data/data_dicts/{dataset_name}.json"
    data_dict = json.load(dict_path.open())

    tar_keys = list(data_dict.keys())
    image_ids = [list(data_dict[key].keys()) for key in data_dict]
    dataset = []
    for i, tar_key in tqdm(enumerate(tar_keys)):
        for img_id in image_ids[i]:
            dataset.append([tar_key, img_id, data_dict[tar_key][img_id][1]])

    # generate captions
    captions = [datum[2] for datum in dataset]
    print(f"Gathered {len(captions)} captions in {time.time() - curr_time:.2f}sec")

    print("---- generate embeddings ----")
    num_gpus = torch.cuda.device_count()
    curr_time = time.time()
    embeddings = embed_captions(captions, encoder, 2048, num_gpus)
    print(f"generated embeddings in {time.time() - curr_time:.2f}sec")

    print("---- filter embeddings ----")
    # filter things with 0 norm
    embed_valid = embeddings.norm(p=2, dim=1) > 0
    dataset = [dataset[i] for i in range(len(dataset)) if embed_valid[i]]
    embeddings = embeddings[embed_valid]
    n_orig = embed_valid.shape[0]
    n_filt = embed_valid.float().sum()
    print(f"{n_orig - n_filt}/{n_orig} captions could not be embedded.")

    return dataset, embeddings


def sample_pairs(dataset, embeddings):
    # sample
    curr_time = time.time()
    embeddings = normalize(embeddings, p=2, dim=1)
    print("Extracting nearest neighbors")
    num_gpus = torch.cuda.device_count()
    distnn, idnn = faiss_knn(
        embeddings, embeddings, k=1, num_gpus=num_gpus, exclude_self=True, pbar=True
    )
    sim_nn = 1 - 0.5 * distnn

    # first two elements are image folder and image id
    data = [datum[:2] for datum in dataset]
    data_nn = [[data[_ij] for _ij in idnn_k] for idnn_k in idnn]
    data_all = [
        (data[i], data_nn[i][0], sim_nn[i, 0].item()) for i in range(len(dataset))
    ]
    print(f"Computed Nearest Neighbors in {time.time() - curr_time:.2f}sec")

    return data_all


def generate_dict(dataset_name, encoder):

    # extract dataset and embeddings
    dataset, embeddings = extract_language_embeddings(dataset_name, encoder)

    # sample pairs
    nn_pairs = sample_pairs(dataset, embeddings)

    # save dict
    save_dir = Path(__file__).parent / "../data/data_dicts"
    save_path = save_dir / f"{dataset_name}_{encoder}.json"

    json.dump(nn_pairs, save_path.open("w"))


if __name__ == "__main__":
    # --- define data ---

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("encoder", type=str)
    args = parser.parse_args()

    generate_dict(args.dataset, args.encoder)
