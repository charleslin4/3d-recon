import os
import argparse

import torch
from torch.utils.data import DataLoader
from pytorch3d.datasets import R2N2, collate_batched_R2N2

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from autoencoder import AutoEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--data_dir', type=str, default='/home/data')
    args = parser.parse_args()

    synsets = ['cabinet']  # use one synset for easy prototyping
    shapenet_path = os.path.join(args.data_dir, 'ShapeNet/ShapeNetCore.v1')
    r2n2_path = os.path.join(args.data_dir, 'ShapeNet')
    r2n2_splits = os.path.join(args.data_dir, 'ShapeNet/ShapeNetRendering/pix2mesh_splits_val05.json')

    train_set = R2N2("train", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False)
    val_set = R2N2("val", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False)

    train_loader = DataLoader(train_set, batch_size=4, collate_fn=collate_batched_R2N2, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=4, collate_fn=collate_batched_R2N2, num_workers=2)

    ae = AutoEncoder()  # TODO pass in whatever args are used in the constructor
    wandb_logger = WandbLogger(name='test', project='3d-recon', entity='3drecon2')
    trainer = pl.Trainer(gpus=args.gpus, logger=wandb_logger)
    trainer.fit(ae, train_loader, val_loader)

