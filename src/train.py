import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch3d.datasets import R2N2, collate_batched_R2N2, BlenderCamera
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PointLights,
    NormWeightedCompositor
)
from data_process import collate_fn

import wandb
from autoencoder import AutoEncoder
from utils import save_checkpoint_model


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def render_points(point_clouds, R, T, K):
    camera = BlenderCamera(device=DEVICE, R=R, T=T, K=K)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=camera, 
            raster_settings=PointsRasterizationSettings(image_size=137)
        ),
        compositor=NormWeightedCompositor()
    )

    images = renderer(point_clouds, cameras=camera, lights=PointLights())  # (N, H, W, C)
    silhouettes = 1. - (1 - images).prod(dim=-1)
    images = images.permute(0, 3, 1, 2)  # (N, C, H, W)
    return images, silhouettes


def train(args):
    shapenet_path = os.path.join(args.data_dir, 'ShapeNet/ShapeNetCore.v1')
    r2n2_path = os.path.join(args.data_dir, 'ShapeNet')
    r2n2_splits = os.path.join('./data/bench_splits.json')

    train_set = R2N2("train", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False, load_textures=False)
    val_set = R2N2("val", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False, load_textures=False)

    ae = AutoEncoder().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-4)
    wandb.watch(ae, log_freq=25)

    total_iters = 0
    for epoch in range(args.epochs):
        train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

        for batch_idx, batch in enumerate(train_loader):
            total_iters += args.bs
            
            opt.zero_grad()
            ae.train()

            images = batch['images']
            batch_size, n_views = images.shape[0], images.shape[1]
            idx = np.asarray([np.random.choice(n_views, 2, replace=False) for i in range(batch_size)])
            input_idx, support_idx = idx[:, 0], idx[:, 1]

            input_images = images[range(batch_size), input_idx].to(DEVICE)      # (N, H, W, C)
            support_images = images[range(batch_size), support_idx].to(DEVICE)  # (N, H, W, C)
            input_masks = 1. - (1 - input_images).prod(dim=-1)
            support_masks = 1. - (1 - support_images).prod(dim=-1)
            input_images = input_images.permute(0, 3, 1, 2)      # (N, C, H, W)
            support_images = support_images.permute(0, 3, 1, 2)  # (N, C, H, W)

            points, textures = ae(input_images)
            point_clouds = Pointclouds(points, features=textures)

            R, T, K = batch['R'], batch['T'], batch['K']
            out_images, out_masks = render_points(
                point_clouds,
                R[range(batch_size), input_idx],
                T[range(batch_size), input_idx],
                K[range(batch_size), input_idx]
            )

            out_support_images, out_support_masks = render_points(
                point_clouds,
                R[range(batch_size), support_idx],
                T[range(batch_size), support_idx],
                K[range(batch_size), support_idx]
            )

            l_base_mask = (
                1. - (input_masks * out_masks).sum(dim=(1,2)) / ((input_masks + out_masks - input_masks * out_masks).sum(dim=(1,2)))
            ).mean()
            l_base_l1 = F.l1_loss(input_images, out_images)
            l_base = l_base_mask + l_base_l1

            l_support_mask = (
                1. - (support_masks * out_support_masks).sum(dim=(1,2)) / ((support_masks + out_support_masks - support_masks * out_support_masks).sum(dim=(1,2)))
            ).mean()
            l_support_l1 = F.l1_loss(support_images, out_support_images)
            l_support = l_support_mask + l_support_l1

            loss = l_base + l_support
            loss.backward()
            opt.step()

            print("Training step {}\tLoss: {:.2f}\tBase mask loss: {:.2f}\tL1 mask loss: {:.2f}\t"
                "Support mask loss: {:.2f}\tSupport L1 loss: {:.2f}".format(
                    batch_idx, loss, l_base_mask, l_base_l1, l_support_mask, l_support_l1
            ))

            wandb.log({
                'loss/train': loss,
                'loss/base_mask': l_base_mask,
                'loss/base_l1': l_base_l1,
                'loss/support_mask': l_support_mask,
                'loss/support_l1': l_support_l1,
                'images/input_gt': wandb.Image(input_images[0]),
                'images/input_recon': wandb.Image(out_images[0]),
                'images/support_gt': wandb.Image(support_images[0]),
                'images/support_recon': wandb.Image(out_support_images[0]),
                'masks/input_gt': wandb.Image(input_masks[0]),
                'masks/input_recon': wandb.Image(out_masks[0]),
                'masks/support_gt': wandb.Image(support_masks[0]),
                'masks/support_recon': wandb.Image(out_support_masks[0]),
                'pt_cloud': wandb.Object3D(point_clouds._points_padded[0].detach().cpu().numpy())
            })
            
            if total_iters % args.save_freq == 0:
                print('Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_checkpoint_model(model, args.model_name, epoch, loss, args.checkpoint_dir, total_iters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--data_dir', type=str, default='/home/data')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='baseline')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/checkpoints')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wandb.init(project='3d-recon', entity='3drecon2')
    train(args)

