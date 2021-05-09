import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch3d.datasets import R2N2, collate_batched_R2N2
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer
from pytorch3d.ops import sample_points_from_meshes

from autoencoder import AutoEncoder
from train import render_points

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
POINT_CLOUD_SIZE = 10000


def compute_min_cd(pc_set, x_gt):
    '''
    Computes the minimum Chamfer Distance (CD) between a ground truth
    point cloud and a set of generated point clouds.

    Input:
        pc_set: a list of Pointcloud objects
        x_gt: a single Pointcloud object

    Returns:
        minimum Chamfer Distance
    '''

    num_samples = pc_set._N
    x_gt_set = x_gt.repeat(num_samples, 1, 1)  # reshape into (N, P, D)
    cds, _ = chamfer.chamfer_distance(pc_set, x_gt_set, batch_reduction=None)
    min_cd = cds.min()
    return min_cd


def compute_mean_iou():
    pass


def evaluate():
    print('============================================================')
    print('============ Evaluate 3D Point Cloud Generation ============')
    print('============================================================')

    # Load test data
    shapenet_path = os.path.join(args.data_dir, 'ShapeNet/ShapeNetCore.v1')
    r2n2_path = os.path.join(args.data_dir, 'ShapeNet')
    r2n2_splits = os.path.join(args.data_dir, 'ShapeNet/ShapeNetRendering/pix2mesh_splits_val05.json')
    test_set = R2N2("val", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False)

    # Set path to save 3D point clouds
    results_dir = os.path.join(args.results_dir, f'{args.model}_{args.name}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize & load models
    if args.model == 'baseline':
        model = AutoEncoder().to(DEVICE)
    elif args.model == 'vq-vae':
        # ADD IMPORT STATEMENT FOR VQ-VAE
        # model = 
        pass
    else:
        raise ValueError("Model [%s] not recognized." % args.model)
    if args.load_dir is not None:
        model.load_state_dict(torch.load(opt.load_dir))

    model.eval()
    
    with torch.no_grad():
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_batched_R2N2)
        total_min_cd = 0.0
        count = 0
        for batch_idx, batch in enumerate(test_loader):
            images = batch['images'].squeeze(0).to(DEVICE)  # (N, H, W, C)
            masks = 1. - (1 - images).prod(dim=-1)
            images = images.permute(0, 3, 1, 2)             # (N, C, H, W)

            n_views = images.shape[1]

            points, textures = model(images)
            point_clouds = Pointclouds(points, textures)

            '''
            R, T, K = batch['R'], batch['T'], batch['K']
            out_images, out_masks = render_points(
                point_clouds,
                R[range(batch_size), input_idx],
                T[range(batch_size), input_idx],
                K[range(batch_size), input_idx]
            )
            '''

            meshes_gt = batch['mesh']
            point_clouds_gt = sample_points_from_meshes(
                meshes_gt,
                num_samples=POINT_CLOUD_SIZE
            ).to(DEVICE)
            min_cd = compute_min_cd(point_clouds, point_clouds_gt)

            # TODO calculate F1 scores
            # TODO periodically save generated point clouds

            print('Step {}\tMinimum CD: {:.2f}'.format(
                    batch_idx, min_cd))
            # TODO write to a log file

            total_min_cd += min_cd
            count += 1

        print(f"Mean Min Chamfer Distance: {total_min_cd / count}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline', help='baseline or vq-vae')
    parser.add_argument('--name', type=str, default='0', help='name of experiment')
    parser.add_argument('--bs', type=int, default=8, help='batch size')
    parser.add_argument('--data_dir', type=str, default='/home/data')

    # Options for testing
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='./eval', help='path to save results to')
    args = parser.parse_args()

    evaluate()
