import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from pytorch3d.datasets import R2N2, collate_batched_R2N2
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer
from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

from autoencoder import AutoEncoder
from train import render_points
from utils import save_point_clouds

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
POINT_CLOUD_SIZE = 10000


def compute_min_cd(pc_set, x_gt):
    '''
    Computes the minimum Chamfer Distance (CD) between a ground truth
    Pointclouds object and a set of generated Pointclouds objects.

    Inputs:
        pc_set: a list of Pointclouds objects
        x_gt: a single Pointclouds object

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

def compute_f1_scores(pred_points, gt_points, threshold=[0.1, 0.2, 0.3, 0.4, 0.5], eps=1e-8):
    '''
    Inputs:
        pred_points: Tensor of shape (N, S, 3) giving coordinates for each predicted Pointcloud
        gt_points: Tensor of shape (N, S, 3) giving coordinates for each ground-truth Pointcloud
        thresholds: The distance thresholds to use when computing F1 scores
        eps: small number to handle numerically instability issues
    
    Returns:
        f1_scores: a dictionary mapping thresholds to F1 scores (scalars)
    '''
    pred_lengths = torch.full((pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device)
    gt_lengths = torch.full((gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device)
    
    knn_pred = knn_points(pred_points, gt_points, lengths1=pred_lengths, lengths2=gt_lengths, K=1)
    knn_gt = knn_points(gt_points, pred_points, lengths1=gt_lengths, lengths2=pred_lengths, K=1)
    
    pred_to_gt_dists = np.sqrt(knn_pred.dists[..., 0]) #(N, S)
    gt_to_pred_dists = np.sqrt(knn_gt.dists[..., 0]) #(N, S)

    f1_scores = {}
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        f1_scores["%f" % t] = f1
    return f1_scores


def evaluate():
    print('============================================================')
    print('============ Evaluate 3D Point Cloud Generation ============')
    print('============================================================')

    # Load test data
    shapenet_path = os.path.join(args.data_dir, 'ShapeNet/ShapeNetCore.v1')
    r2n2_path = os.path.join(args.data_dir, 'ShapeNet')
    r2n2_splits = os.path.join(args.data_dir, 'ShapeNet/ShapeNetRendering/pix2mesh_splits_val05.json')
    test_set = R2N2("val", shapenet_path, r2n2_path, r2n2_splits, return_voxels=False)

    # Save path for metrics and point clouds
    results_dir = os.path.join(args.results_dir, f'{args.model}_{args.name}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print("Point clouds will be saved in: " + results_dir)
    test_metrics_file = os.path.join(results_dir, 'test_metrics.csv')
    print("Test metrics will be written to: " + test_metrics_file)
    
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
        checkpoint = torch.load(opt.load_dir)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    
    with torch.no_grad():
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_batched_R2N2)
        total_min_cd = 0.0
        count = 0
        num_batches = len(test_loader) // args.bs
        batch_idx_min_cd = np.zeros((num_batches, 2))
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
            points_gt = sample_points_from_meshes(
                meshes_gt,
                num_samples=POINT_CLOUD_SIZE
            ).to(DEVICE)
            
            # Compute metrics (minimum Chamfer distance and F1 scores)
            min_cd = compute_min_cd(point_clouds, points_gt)
            f1_scores = compute_f1_scores(points, points_gt)

            # Renderer settings to save generated point cloud images
            R, T = look_at_view_transform(20, 10, 0)
            cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
            raster_settings = PointsRasterizationSettings(
                image_size=512, 
                radius = 0.003,
                points_per_pixel = 10
            )
            
            renderer = PointsRenderer(
                rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
                compositor=NormWeightedCompositor()
            )

            # Periodically save generated point clouds objects and their rendered images
            for j in range(len(points)):
                obj_id = batch_idx * args.bs + j
                if obj_id % args.save_freq == 0:
                    pc_obj_file = os.path.join(results_dir, '%06d.pth' % obj_id)
                    pc_img_file = os.path.join(results_dir, '%06d.png' % obj_id)
                    pc_gt_img_file = os.path.join(results_dir, '%06d_gt.png' % obj_id)
                    
                    point_clouds_j = Pointclouds(points[j,:,:,:], textures[j,:,:,:])
                    images = renderer(point_clouds_j)
                    img = images[0, ..., :3].cpu().numpy()
                    point_clouds_gt = Pointclouds(points_gt[j,:,:,:], batch['textures'][j,:,:,:])
                    images_gt = renderer(point_clouds_gt)
                    img_gt = images_gt[0, ..., :3].cpu().numpy()

                    save_point_clouds(pc_obj_file, pc_img_file, pc_gt_img_file, points[j,:,:,:], points_gt[j,:,:,:], img, img_gt)
            
            # Print and save metrics to .csv file
            print('Step {}\tMinimum CD: {:.2f}'.format(
                    batch_idx, min_cd))
            batch_idx_min_cd[count][0] = batch_idx
            batch_idx_min_cd[count][1] = min_cd

            total_min_cd += min_cd
            count += 1

        mean_min_cd = total_min_cd / count
        print(f"Mean Min Chamfer Distance: {mean_min_cd}")
        print(f"F1 Scores for thresholds [0.1, 0.2, 0.3, 0.4, 0.4]: {f1_scores.items()}")
        batch_idx_min_cd[0][2] = mean_min_cd
        metrics = pd.DataFrame(batch_idx_min_cd, columns=['Batch Index', 'Minimum CD', 'Overall Mean Min CD']).to_csv(test_metrics_file)


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
