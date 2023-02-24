Learning Visual Representations via Language-Guided Sampling
====================================

<img src="https://user-images.githubusercontent.com/11287599/221066639-9553cb09-a1c7-4bed-99dc-21cd14250215.png" width="50%">

**[Learning Visual Representations via Language-Guided Sampling][1]**  
[Mohamed El Banani][2], [Karan Desai][3], and [Justin Johnson][4]

If you have any questions, please feel free to email me at [mbanani@umich.edu](mailto:mbanani@umich.edu).



Environment Setup
-----------------

We recommend using Anaconda or Miniconda. To setup the environment, follow the instructions below.

```bash
conda create -n lgssl python=3.8 --yes
conda activate lgssl
conda install pytorch=1.12.1 torchvision cudatoolkit=11.3 -c pytorch --yes

python -m pip install -r requirements.txt
python setup.py develop
```

## Training Datasets 
<details>
<summary> Expand </summary>
<p>

We train our models on RedCaps and ConceptualCaptions (CC3M and CC12M). 
We note that all 3 datasets can decay, so you might end up with a different number of instances. 
Please refer to the original papers for dataset download instructions. 
In our case, the datasets had the following sizes:

| Dataset      | Size        |
| -----------  | ----------- |
| RedCaps-2020 |    3273223  | 
| RedCaps      |   12010494  |
| CC3M         |    2913035  |
| CC12M        |   10958691  |


We assume all training datasets are in `data/datasets` which is set as the default `data_root` in
the base [dataset class](./lgssl/datasets/base_dataset.py). We expect the dataset to be in the format
below where each dataset is subdivided into several directories and each directory contains a set of
instances where each instance has an image file and a json caption file. 

```
data/datasets/<dataset_name>
    |- directory_0
        |- <instance_0>.jpg     <- image for instance 0
        |- <instance_0>.json    <- caption for instance 0
        |- <instance_1>.jpg
        |- <instance_1>.json
        ...
        |- <instance_n>.jpg
        |- <instance_n>.json
    |- directory_1
    |- directory_2
    ...
    |- directory_m
```

For RedCaps, the directory names are encoded as `<subreddit>_<year>_<id>`, e.g.,
`crochet_2017_000001`, where each directory only has 10000 classes. We use this naming convention
for some of the experiments: experiments with redcaps-2020 and sampling scope. 

### Generating dataset dictionaries

Coming soon.

### Sampling nearest neighbor pairs

Coming soon.

  </p>
</details>

## Evaluation Datasets

<details>
<summary> Expand </summary>
<p>
We use [TensorFlow Datasets](https://www.tensorflow.org/datasets) for our evaluations. 
This package provides us with all the evaluations except for FGVC Aircraft. 
Our code will automatically download and extract all the
datasets in `data/evaluation_datasets` on the first run of the evaluation code.
This means that the first evaluation run will be much slower than usual.  

**Note 1:** We encountered a bug with SUN 397 where one image could not be decoded correctly. 
This is a known [bug](https://github.com/tensorflow/datasets/pull/3951) which has not been fixed yet
in the stable version. To fix it, simply make the two changes outlined by this
[commit](https://github.com/tensorflow/datasets/pull/3951/commits/c4ff599a357ee92f5f0584efb715939299f1d13e).

**Note 2:**
TensorFlow Datasets will require you to independently downloaded RESISC45. Please follow the
instructions provided [here](https://www.tensorflow.org/datasets/catalog/resisc45)
</p>
</details>

Training models
----------------

We use hydra configs for our training experiments. The configs can all be found [here](./configs/).
To run an experiment, you can either to define a new [experiment config](./configs/experiment/) which can be used to override the [default configs](./configs/train.yaml). 
Alternatively, you can just overwrite some configs in the command.
We provide a few sample training commands configs for clarity:

```bash
python train.py +experiment=ours                        % LG SimCLR
python train.py +experiment=vis_baseline                % SimCLR 
python train.py +experiment=vis_baseline model=simsiam  % SimSiam
```


Evaluation
-----------

We use two primary evaluations: linear probe using L-BFGS and few-shot evaluation. The configs for those evaluations can be found [here](./configs/evaluation.yaml). 

**Linear Probe**: we train a single layer using logistic regression and sweep over regualizer weight values. 
We provide an [implementation](./lgssl/evaluation/logistic_regression.py) of logistic regression using PyTorch's L-BFGS, however, you can easily use scikit-learn's implementation by setting the `use_sklearn` flag in the [evaluation configs](./configs/evaluation.yaml).
For datasets without a standard validation split, we randomly split the training set while maintaining the class distribution. 

**Few-Shot Evaluation**: we also evaluate our frozen features on 5-shot, 5-way classification. The evaluation can be found [here](./lgssl/evaluation/fewshot.py).
We sample the training samples from the train/valid splits and the query samples for the test set. 

The following commands can be used to evaluate checkpoints or baselines. For example, you can
evaluate our model or the pretrained SimCLR checkpoint on all the datasets by running the following commands: 

```bash
python evaluate.py model.name=lgssl_checkpoints model.checkpoint=lgsimclr dataset.name=all
python evaluate.py model.name=simclr dataset.name=all
```

Pre-trained Checkpoints
------------------------

You can find all our pretrained checkpoints
[here](https://www.dropbox.com/sh/me6nyiewlux1yh8/AAAPrD2G0_q_ZwExsVOS_jHQa?dl=0). You should
download them to `data/checkpoints`. More details coming soon.

## Citation 

If you find this code useful, please consider citing:  

```bibtex
@inProceedings{elbanani2022languageguided,
  title={{Learning Visual Representations via Language-Guided Sampling}},
  author={El Banani, Mohamed and Desai, Karan and Johnson, Justin},
  year={2022},
}
```

Acknowledgments
---------------
We thank Richard Higgins, Ashkan Kazemi, and Santiago Castro for many helpful discussions.
We also thank David Fouhey, Ziyang Chen, Chenhao Zheng, and Fahad Kamran, and Dandan Shan for their feedback on early drafts. 
This project was funded under the Ford-UM Alliance partnership. 
We thank Alireza Rahimpour, Davesh Upadhyay, and Ali Hassani from Ford Research for their support and discussion.

[1]: https://arxiv.org/abs/2302.12248
[2]: https://mbanani.github.io
[3]: http://kdexd.xyz
[4]: https://web.eecs.umich.edu/~justincj/
