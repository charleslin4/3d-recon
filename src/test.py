import os
import argparse
import numpy as np
import csv
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from pytorch3d.datasets import R2N2, collate_batched_R2N2
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes, knn_points

from dataloader import build_data_loader
from pointalign import PointAlign
from utils import save_point_clouds, format_image, rotate_verts

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
    cds, _ = chamfer_distance(pc_set, x_gt_set, batch_reduction=None)
    min_cd = cds.min()
    return min_cd


def compute_mean_iou():
    pass


def compute_f1_scores(pred_points, gt_points, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], eps=1e-8):
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
    
    pred_to_gt_dists = np.sqrt(knn_pred.dists[..., 0].cpu()) #(N, S)
    gt_to_pred_dists = np.sqrt(knn_gt.dists[..., 0].cpu()) #(N, S)

    f1_scores = {}
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        f1_scores[t] = f1.sum(axis=0).int()
    return f1_scores

def compute_pc_metrics(pred_points, gt_points, thresholds, eps):
    """
    Compute metrics that are based on sampling points and normals:
    - L2 Chamfer distance
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation
    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
    """
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    metrics["Chamfer-L2"] = chamfer_l2

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics


def get_metrics(pred_points, gt_points, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], eps=1e-8):
    '''
    if torch.is_tensor(pred_points) and torch.is_tensor(gt_points):
        metrics = compute_pc_metrics(
            pred_points, gt_points, thresholds, eps
        )
    else:
        metrics = defaultdict(list)
        for cur_points_pred, cur_points_gt in zip(pred_points, gt_points):
            cur_metrics = compute_pc_metrics(
                cur_points_pred[None], cur_points_gt[None], thresholds, eps
            )
            for k, v in cur_metrics.items():
                metrics[k].append(v.item())
        metrics = {k: torch.tensor(vs) for k, vs in metrics.items()}
    return metrics
    '''
    metrics = defaultdict(list)
    for cur_points_pred, cur_points_gt in zip(pred_points, gt_points):
        cur_metrics = compute_pc_metrics(
            cur_points_pred[None], cur_points_gt[None], thresholds, eps
        )
        for k, v in cur_metrics.items():
            metrics[k].append(v.item())
    metrics = {k: torch.tensor(vs) for k, vs in metrics.items()}
    return metrics


def evaluate():
    print('============================================================')
    print('============ Evaluate 3D Point Cloud Generation ============')
    print('============================================================')

    # Load test data
    test_loader = build_data_loader(
        data_dir=args.data_dir,
        split_name='test',
        splits_file=args.splits_path,
        batch_size=args.bs,
        num_workers=4,
        multigpu=False,
        shuffle=True,
        num_samples=None
    )

    class_names = {
        "02828884": "bench",
        "03001627": "chair",
        "03636649": "lamp",
        "03691459": "speaker",
        "04090263": "firearm",
        "04379243": "table",
        "04530566": "watercraft",
        "02691156": "plane",
        "02933112": "cabinet",
        "02958343": "car",
        "03211117": "monitor",
        "04256520": "couch",
        "04401088": "cellphone",
    }
    num_instances = {i: 0 for i in class_names}
    chamfer = {i: 0 for i in class_names}
    f1_01 = {i: 0 for i in class_names}
    f1_03 = {i: 0 for i in class_names}
    f1_05 = {i: 0 for i in class_names}
    count = 0

    # Save path for metrics and point clouds
    results_dir = os.path.join(args.results_dir, f'{args.model_name}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    metrics_file = os.path.join(results_dir, 'metrics.csv')
    print("\nPoint clouds will be saved in: " + results_dir)
    print("Metrics (Chamfer distance and F1 scores) will be written to: " + metrics_file)
    
    # Initialize & load models
    if args.model_name == 'baseline':
        model = PointAlign().to(DEVICE)
    elif args.model_name == 'vq-vae':
        # ADD IMPORT STATEMENT FOR VQ-VAE
        # model = 
        pass
    else:
        raise ValueError("Model [%s] not recognized." % args.model)
    if args.load_dir is not None:
        checkpoint = torch.load(opt.checkpoint_dir)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, _, ptclds_gt, normals, RT, K, id_strs = test_loader.postprocess(batch)
            #sids = [id_str.split("-")[0] for id_str in id_strs]
            sids = [id_strs[0].split("-")[0]]
            for sid in sids:
                num_instances[sid] += 1
            
            ptclds_pred, textures_pred = model(images)
            ptclds_gt_cam = rotate_verts(RT, ptclds_gt) # do some rotation here
            cur_metrics = get_metrics(ptclds_pred, ptclds_gt)

            for i, sid in enumerate(sids):
                chamfer[sid] += cur_metrics["Chamfer-L2"][i].item()
                f1_01[sid] += cur_metrics["F1@%f" % 0.1][i].item()
                f1_03[sid] += cur_metrics["F1@%f" % 0.3][i].item()
                f1_05[sid] += cur_metrics["F1@%f" % 0.5][i].item()

                # Periodically save generated & ground-truth pointcloud objects and their rendered images
                if i % args.save_freq == 0:
                    save_point_clouds(id_strs[i], ptclds_pred, ptclds_gt, results_dir)

        col_headers = class_names.values()
        chamfer_save = {class_names[key] : value for key, value in chamfer.items()}
        f1_01_save = {class_names[key] : value for key, value in f1_01.items()}
        f1_03_save = {class_names[key] : value for key, value in f1_03.items()}
        f1_05_save = {class_names[key] : value for key, value in f1_05.items()}
        metrics = [chamfer_save, f1_01_save, f1_03_save, f1_05_save] 
        with open(metrics_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=col_headers)
            writer.writeheader()
            writer.writerows(metrics)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='baseline', help='baseline or vq-vae')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--data_dir', type=str, default='/home/data/ShapeNet/ShapeNetV1processed')
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--checkpoint_dir', type=str, default='/home/checkpoints')
    parser.add_argument('--splits_path', type=str, default='./data/bench_splits.json')

    # Options for testing
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='./results', help='path to save results to')
    args = parser.parse_args()

    evaluate()
