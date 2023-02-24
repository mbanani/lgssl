# Evaluation Baselines 

We compare our model against several publicly available checkpoints. While some checkpoints are
hosted by some repositories, others require manual download. Below are the baselines we use and
links to manually download them. 
We also provide a script `download_baselines.sh` to download all the weights. 


| Baseline                          | Checkpoint URL         | 
| -----------                       | -----------            |
| SimCLR (PyTorch Lightning Bolts)  | [link](https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt) |
| SimSiam                           | [link](https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar) |
| SwAV                              | [link](https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar) | 
| MoCo v3                           | [link](https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/r-50-100ep.pth.tar) |
