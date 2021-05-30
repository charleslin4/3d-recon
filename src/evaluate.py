import os
import random
import argparse
import numpy as np
import csv
import json

import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

from omegaconf import DictConfig, OmegaConf
from hydra.experimental import compose, initialize_config_dir

from datautils.dataloader import build_data_loader
from models.pointalign import PointAlign, PointAlignSmall
import utils


torch.multiprocessing.set_sharing_strategy('file_system')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    cham_dist, _ = chamfer_distance(ptclds1, ptclds2, batch_reduction=None, point_reduction="mean")
    metrics["Chamfer"] = cham_dist
    
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
        metrics["F1@%f" % t] = f1

    return


def compute_metrics(ptclds_pred, ptclds_gt, thresholds):
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
    compute_f1_scores(metrics, ptclds_pred, ptclds_gt, thresholds)
    metrics = {k: v.cpu() for k, v in metrics.items()}

    return metrics


def render_ptcld(ptcld, P, save_file):
    '''
    Renders a single 3D point cloud.

    Inputs:
        ptcld: Tensor of shape (N, S, 3)
        P: rotation matrix
        save_file: file to save render to
    
    Returns:
        ptcld_img: 2D rendered image of 3D point cloud in camera coordinates
    '''

    ptcld_rot = utils.rotate_verts(P, ptcld).cpu()
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(ptclds[..., 0], ptclds[..., 2], ptclds[..., 1], s=1)
    ax.view_init(elev=0, azim=270)
    ax.grid(False)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.axis('off')

    plt.savefig(save_file, bbox_inches="tight")


def evaluate(config: DictConfig):
    '''
    Tests the model and saves metrics and point cloud predictions.

    Adapted from https://github.com/facebookresearch/meshrcnn/blob/df9617e9089f8d3454be092261eead3ca48abc29/shapenet/evaluation/eval.py
    '''

    orig_dir = os.getcwd()
    results_dir = os.path.join(config.run_dir, 'results')  # Save checkpoints to run directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    metrics_file = os.path.join(results_dir, 'metrics.csv')
    print("Point clouds will be saved in: " + results_dir)
    print("Metrics will be written to: " + metrics_file)

    model_name = config.model
    splits_path = os.path.join(orig_dir, config.splits_path)  # Load splits from original path
    test_loader = build_data_loader(
        data_dir=config.data_dir,
        split_name='test',
        splits_file=splits_path,
        batch_size=config.bs,
        num_workers=4,
        multigpu=False,
        shuffle=True,
        num_samples=None
    )

    model = utils.load_model(model_name, config)
    model.to(DEVICE)
    model.eval()

    splits = json.load(open(splits_path, "r"))
    syn_ids = [syn_id for syn_id in splits['test'].keys()]

    num_instances = {syn_id: 0 for syn_id in syn_ids}
    chamfer = {syn_id: 0 for syn_id in syn_ids}
    f1_1e_4 = {syn_id: 0 for syn_id in syn_ids}
    f1_2e_4 = {syn_id: 0 for syn_id in syn_ids}

    print("Starting evaluation")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, _, ptclds_gt, normals, RT, K, id_strs = test_loader.postprocess(batch)
            sids = [id_str.split("-")[0] for id_str in id_strs]
            for sid in sids:
                num_instances[sid] += 1

            if model_name == 'vqvae':
                ptclds_pred, l_vq, encoding_inds, perplexity = model(images, RT)
            elif model_name == 'vae':
                ptclds_pred, mu, logvar = model(images, RT)
            else:
                ptclds_pred = model(images, RT)
            cur_metrics = compute_metrics(ptclds_pred, ptclds_gt, thresholds=[0.01, 0.014142])

            for i, sid in enumerate(sids):
                chamfer[sid] += cur_metrics['Chamfer'][i].item()
                f1_1e_4[sid] += cur_metrics["F1@%f" % 0.01][i].item()
                f1_2e_4[sid] += cur_metrics["F1@%f" % 0.014142][i].item()

            if batch_idx % config.save_freq == 0:
                for i, sid in enumerate(sid):
                    utils.save_point_clouds(id_strs[i] + str(batch_idx), ptclds_pred, ptclds_gt, results_dir)

        col_headers = [class_names[syn_id] for syn_id in syn_ids]
        chamfer_final = {class_names[syn_id] : cham_dist / num_instances[syn_id] for syn_id, cham_dist in chamfer.items()}
        f1_1e_4_final = {class_names[syn_id] : f1_score / num_instances[syn_id] for syn_id, f1_score in f1_1e_4.items()}
        f1_2e_4_final = {class_names[syn_id] : f1_score / num_instances[syn_id] for syn_id, f1_score in f1_2e_4.items()}
        metrics = [chamfer_final, f1_1e_4_final, f1_2e_4_final] 
        with open(metrics_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=col_headers)
            writer.writeheader()
            writer.writerows(metrics)

    print(f"Saved evaluation metrics to {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--checkpoint_path', default=None, help='(Relative or absolute) path of the model checkpoint')
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path) # /path/to/run_dir/checkpoints/checkpoint.pth
    run_dir = os.path.dirname(checkpoints_dir)

    config_dir = os.path.join(run_dir, '.hydra')
    initialize_config_dir(config_dir=config_dir, job_name='evaluate')
    overrides = [
        f"+run_dir={run_dir}",
        f"+checkpoint_path={checkpoint_path}"
    ]
    config = compose(config_name='config', overrides=overrides)
    print(OmegaConf.to_yaml(config))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    evaluate(config)

