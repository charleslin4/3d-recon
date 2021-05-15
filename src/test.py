import os
import argparse
import numpy as np
import csv
import pdb, traceback, sys

import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

from dataloader import build_data_loader
from pointalign import PointAlign
from utils import save_point_clouds, format_image, rotate_verts, imagenet_deprocess

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
POINT_CLOUD_SIZE = 10000
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


def compute_min_cham_dist(ptclds_set, ptcld_gt):
    '''
    Computes the minimum Chamfer Distance between a ground truth
    Pointclouds object and a set of generated Pointclouds objects.

    Inputs:
        ptclds_set: a list of Pointclouds objects
        ptcld_gt: a single Pointclouds object

    Returns:
        min_cham_dist: minimum Chamfer Distance
    '''

    num_samples = ptclds_set._N
    ptclds_gt = ptcld_gt.repeat(num_samples, 1, 1)  # reshape into (N, P, D)
    cds, _ = chamfer_distance(ptclds_set, ptclds_gt, batch_reduction=None)
    min_cd = cds.min()
    return min_cd


def compute_cham_dist(metrics, ptclds1, ptclds2):
    '''
    Computes the sum of the mean Chamfer Distance between two batches of point clouds.
    Stores the batch sum of Chamfer Distances in metrics.

    Inputs:
        metrics: dict storing all test metrics
        ptclds1: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with at most P1 points in each batch element, batch size N, and feature dimension D.
        ptclds2: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with at most P2 points in each batch element, batch size N, and feature dimension D.
    '''
    
    cham_dist = chamfer_distance(ptclds1, ptclds2, batch_reduction=None, point_reduction="mean")
    metrics["Chamfer"] = cham_dist[0].sum(axis=0)
    
    return


def compute_f1_scores(metrics, ptclds_pred, ptclds_gt, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], eps=1e-8):
    '''
    Computes F1 scores between each pair of predicted and ground-truth point clouds at the given thresholds.
    Stores the batch sum of F1 scores in metrics.

    Inputs:
        ptclds_pred: Tensor of shape (N, S, 3) giving coordinates for each predicted point cloud
        ptclds_gt: Tensor of shape (N, S, 3) giving coordinates for each ground-truth point cloud
        thresholds: distance thresholds to use when computing F1 scores
        eps: small number to handle numerically instability issues

    Returns:
        metrics: dict storing all test metrics
    
    Adapted from https://github.com/facebookresearch/meshrcnn/blob/df9617e9089f8d3454be092261eead3ca48abc29/meshrcnn/utils/metrics.py
    '''

    pred_lengths = torch.full((ptclds_pred.shape[0],), ptclds_pred.shape[1], dtype=torch.int64, device=ptclds_pred.device)
    gt_lengths = torch.full((ptclds_gt.shape[0],), ptclds_gt.shape[1], dtype=torch.int64, device=ptclds_gt.device)
    
    # For each predicted point, find its neareast-neighbor ground-truth point
    knn_pred = knn_points(ptclds_pred, ptclds_gt, lengths1=pred_lengths, lengths2=gt_lengths, K=1)
    # Compute L2 distances between each predicted point and its nearest ground-truth
    pred_to_gt_dists = np.sqrt(knn_pred.dists[..., 0].cpu()) #(N, S)
    
    # For each ground-truth point, find its nearest-neighbor predicted point
    knn_gt = knn_points(ptclds_gt, ptclds_pred, lengths1=gt_lengths, lengths2=pred_lengths, K=1)
    # Compute L2 distances between each ground-truth point and its nearest predicted point
    gt_to_pred_dists = np.sqrt(knn_gt.dists[..., 0].cpu()) #(N, S)

    # Compute F1 scores based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1) #(N,)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1) #(N,)
        f1 = (2.0 * precision * recall) / (precision + recall + eps) #(N,)
        metrics["F1@%f" % t] = f1.sum(axis=0)
    
    return


def compute_mean_iou():
    pass


def compute_metrics(ptclds_pred, ptclds_gt):
    '''
    Inputs:
        ptclds_pred: Tensor of shape (N, S, 3) giving coordinates for each predicted point cloud
        ptclds_gt: Tensor of shape (N, S, 3) giving coordinates for each ground-truth point cloud
        thresholds: distance thresholds to use when computing F1 scores
        eps: small number to handle numerically instability issues


    Returns:
        metrics:

    Adapted from https://github.com/facebookresearch/meshrcnn/blob/df9617e9089f8d3454be092261eead3ca48abc29/meshrcnn/utils/metrics.py
    '''
    metrics = {}
    compute_cham_dist(metrics, ptclds_pred, ptclds_gt)
    compute_f1_scores(metrics, ptclds_pred, ptclds_gt)
    metrics = {k: v.cpu() for k, v in metrics.items()}

    return metrics

# TODO render point clouds (see render_point_clouds notebook)
def render_ptcld(ptcld):
    '''
    Renders a single 3D point cloud.

    Inputs:
        ptcld: Tensor of shape (N, S, 3)
    
    Returns:
        ptcld_img: 2D rendered image of 3D point cloud in camera coordinates
    '''

    return


def evaluate():
    '''
    Tests the model and saves metrics and point cloud predictions.

    Adapted from https://github.com/facebookresearch/meshrcnn/blob/df9617e9089f8d3454be092261eead3ca48abc29/shapenet/evaluation/eval.py
    '''

    results_dir = os.path.join(args.results_dir, f'{args.model_name}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    metrics_file = os.path.join(results_dir, 'metrics.csv')
    print("\nPoint clouds will be saved in: " + results_dir)
    print("Metrics will be written to: " + metrics_file)

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

    if args.model_name == 'baseline':
        model = PointAlign().to(DEVICE)
    elif args.model_name == 'vq-vae':
        # model = 
        pass
    else:
        raise ValueError("Model [%s] not recognized." % args.model)
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if not args.small_dataset:
        syn_ids = [syn_id for syn_id in class_names.keys()]
    else:
        syn_ids = ['02828884']
    num_instances = {syn_id: 0 for syn_id in syn_ids}
    chamfer = {syn_id: 0 for syn_id in syn_ids}
    f1_01 = {syn_id: 0 for syn_id in syn_ids}
    f1_03 = {syn_id: 0 for syn_id in syn_ids}
    f1_05 = {syn_id: 0 for syn_id in syn_ids}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, _, ptclds_gt, normals, RT, K, id_strs = test_loader.postprocess(batch)
            for syn_id in syn_ids:
                num_instances[syn_id] += len(images)
            
            ptclds_pred = model(images)
            ptclds_gt_cam = rotate_verts(RT, ptclds_gt)
            cur_metrics = compute_metrics(ptclds_pred, ptclds_gt_cam) # compute metrics for gt in camera coordinates?

            for i, syn_id in enumerate(syn_ids):
                chamfer[syn_id] += cur_metrics['Chamfer'].item()
                f1_01[syn_id] += cur_metrics['F1@%f' % 0.1].item()
                f1_03[syn_id] += cur_metrics['F1@%f' % 0.3].item()
                f1_05[syn_id] += cur_metrics['F1@%f' % 0.5].item()

            # TODO periodically save rendered point clouds
            #if num_test % args.save_freq == 0:
                #for i, sid in enumerate(sids):
                    #save_point_clouds(id_strs[i] + str(num_test), ptclds_pred, ptclds_gt_cam, results_dir)
    
        col_headers = class_names.values()
        chamfer_final = {class_names[syn_id] : cham_dist/num_instances[syn_id] for syn_id, cham_dist in chamfer.items()}
        f1_01_final = {class_names[syn_id] : f1_01/num_instances[syn_id] for syn_id, f1_01 in f1_01.items()}
        f1_03_final = {class_names[syn_id] : f1_03/num_instances[syn_id] for syn_id, f1_03 in f1_03.items()}
        f1_05_final = {class_names[syn_id] : f1_05/num_instances[syn_id] for syn_id, f1_05 in f1_05.items()}
        metrics = [chamfer_final, f1_01_final, f1_03_final, f1_05_final] 
        with open(metrics_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=col_headers)
            writer.writeheader()
            writer.writerows(metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='baseline', help='baseline or vq-vae')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--data_dir', type=str, default='/home/data/ShapeNet/ShapeNetV1processed')
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--checkpoint_dir', type=str, default='/home/clin/cs231n-project/outputs/2021-05-15/21-07-32/checkpoints/')
    parser.add_argument('--checkpoint_name', type=str, default='PointAlign_2021-05-15.21-05-32_model_0.pth')
    parser.add_argument('--splits_path', type=str, default='./data/bench_splits.json')
    parser.add_argument('--small_dataset', type=bool, default=True)

    # Options for testing
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='./results', help='path to save results to')
    args = parser.parse_args()

    evaluate()
