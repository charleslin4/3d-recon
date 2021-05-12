import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pytorch3d.datasets import R2N2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def collate_fn(batch):
    if batch is None or len(batch) == 0:
        return None
    collated_dict = {}
    for k in batch[0].keys():
        collated_dict[k] = [d[k] for d in batch]

    # If collate_batched_meshes receives R2N2 items with images and that
    # all models have the same number of views V, stack the batches of
    # views of each model into a new batch of shape (N, V, H, W, 3).
    # Otherwise leave it as a list.
    if "images" in collated_dict:
        try:
            collated_dict["images"] = torch.stack(collated_dict["images"])
        except RuntimeError:
            print(
                "Models don't have the same number of views. Now returning "
                "lists of images instead of batches."
            )

    # If collate_batched_meshes receives R2N2 items with camera calibration
    # matrices and that all models have the same number of views V, stack each
    # type of matrices into a new batch of shape (N, V, ...).
    # Otherwise leave them as lists.
    if all(x in collated_dict for x in ["R", "T", "K"]):
        try:
            collated_dict["R"] = torch.stack(collated_dict["R"])  # (N, V, 3, 3)
            collated_dict["T"] = torch.stack(collated_dict["T"])  # (N, V, 3)
            collated_dict["K"] = torch.stack(collated_dict["K"])  # (N, V, 4, 4)
        except RuntimeError:
            print(
                "Models don't have the same number of views. Now returning "
                "lists of calibration matrices instead of a batched tensor."
            )
    
    return collated_dict


def test_collate_fn(r2n2):
    dataloader = DataLoader(r2n2, batch_size=8, shuffle=True, collate_fn=collate_fn)
    for batch_idx, batch in enumerate(dataloader):
        print(batch_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=321)
    parser.add_argument('--data_dir', type=str, default='/home/data')
    parser.add_argument('--bs', default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    shapenet_path = os.path.join(args.data_dir, 'ShapeNet/ShapeNetCore.v1')
    r2n2_path = os.path.join(args.data_dir, 'ShapeNet')
    r2n2_splits = os.path.join('./data/bench_splits.json')

    train_set = R2N2("train", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False, load_textures=False)
    test_collate_fn(train_set)

    val_set = R2N2("val", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False, load_textures=False)
    test_collate_fn(val_set)

    test_set = R2N2("val", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False, load_textures=False)
    test_collate_fn(test_set)


