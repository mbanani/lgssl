wget https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt
mv simclr_imagenet.ckpt simclr.ckpt

wget https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar
mv checkpoint_0099.pth.tar simsiam.pth

wget https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar
mv swav_100ep_pretrain.pth.tar swav.pth

wget https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/r-50-100ep.pth.tar
mv r-50-100ep.pth.tar mocov3.pth
