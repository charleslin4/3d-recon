# 3d-recon

## Dependencies
* `torch==1.7.1`
* `torchvision==0.8.2`
* `pytorch-lightning==1.2.7`
* `pytorch3d==0.4.0`
* `wandb`


After installing Pytorch3D, run the following:

```
wget https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/pytorch3d/datasets/shapenet/shapenet_synset_dict_v1.json -P /path/to/pytorch3d/datasets/shapenet
wget https://raw.githubusercontent.com/facebookresearch/pytorch3d/master/pytorch3d/datasets/r2n2/r2n2_synset_dict.json -P /path/to/pytorch3d/datasets/r2n2
wget https://dl.fbaipublicfiles.com/meshrcnn/shapenet/pix2mesh_splits_val05.json -P /path/to/ShapeNetRendering
```
