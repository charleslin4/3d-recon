import os
import argparse
import utils
import options
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch3d.datasets import R2N2, collate_batched_R2N2
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer
from pytorch3d.ops import sample_points_from_meshes
import wandb

from autoencoder import AutoEncoder
from options import *
from train import render_points

def compute_min_cd(x_gt, pc_set):
    '''
    Computes the minimum Chamfer Distance (CD) between a ground truth
    point cloud and a set of generated point clouds.

    Input:
        x_gt: a single Pointcloud object
        pc_set: list of 24 Pointcloud objects

    Returns:
        minimum Chamfer Distance
    '''
    cds = []
    for x in pc_set:
        cd = chamfer.chamfer_distance(x_gt, x)[0]
        cds.append(cd[0].item())
    return np.min(cds)

def compute_mean_iou():
    pass

def test():
    print('===========================================================')
    print('============ Evaluate 3D Point Cloud Generation ===========')
    print('===========================================================')

    # Parse arguments
    args = options.get_arguments()

    # Load test data
    shapenet_path = os.path.join(args.data_dir, 'ShapeNet/ShapeNetCore.v1')
    r2n2_path = os.path.join(args.data_dir, 'ShapeNet')
    r2n2_splits = os.path.join(args.data_dir, 'ShapeNet/ShapeNetRendering/pix2mesh_splits_val05.json')
    test_set = R2N2("val", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False)

    # Set path to save 3D point clouds
    results_dir = args.results_dir + '/{args.model}_{args.name}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize & load models
    if args.model == 'baseline':
        from .autoencoder import AutoEncoder
        model = AutoEncoder().to(DEVICE) # add AE arguments here
    elif args.model == 'vq-vae':
        # ADD IMPORT STATEMENT FOR VQ-VAE
        # model = 
    else:
        raise ValueError("Model [%s] not recognized." % args.model)
    if args.load_dir is not None:
        model.load_state_dict(torch.load(opt.load_dir))

    model.eval()
    
    with torch.no_grad():
        test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False, collate_fn=collate_batched_R2N2)
        min_cds = np.zeros(len(test_loader), args.bs)
        for batch_idx, batch in enumerate(test_loader):
            images = batch['images']
            batch_size, n_views = images.shape[0], images.shape[1]
            idx = np.asarray([np.random.choice(n_views, 2, replace=False) for i in range(batch_size)])
            input_idx, support_idx = idx[:, 0], idx[:, 1]

            input_images = images[range(batch_size), input_idx].to(DEVICE)      # (N, H, W, C)
            input_masks = 1. - (1 - input_images).prod(dim=-1)
            input_images = input_images.permute(0, 3, 1, 2)      # (N, C, H, W)

            points, textures = model(input_images)
            point_clouds = Pointclouds(points, features=textures)

            R, T, K = batch['R'], batch['T'], batch['K']
            out_images, out_masks = render_points(
                point_clouds,
                R[range(batch_size), input_idx],
                T[range(batch_size), input_idx],
                K[range(batch_size), input_idx]
            )

            meshes_gt = batch['mesh']
            for i in range(batch_size):
                point_cloud_gt = sample_points_from_meshes(
                    meshes_gt[i],
                    num_samples=POINT_CLOUD_SIZE
                )
                min_cds[batch_idx, i] = compute_min_cd(point_clouds, point_cloud_gt)
    
            wandb.log({
                'images/input_gt': wandb.Image(input_images[0]),
                'images/input_recon': wandb.Image(out_images[0]),
                'pt_cloud': wandb.Object3D(point_clouds._points_padded[0].detach().cpu().numpy())
                })
        print("Mean Min Chamfer Distance: %f", min_cds.mean())

if __name__ == "__main__":
    wandb.init(project='3d-recon', entity='3drecon2')
    test()