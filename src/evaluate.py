import os
import random
import argparse
import numpy as np
import csv
import json
import hydra

import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
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
    
    cham_dist, _ = chamfer_distance(ptclds1, ptclds2, batch_reduction='sum', point_reduction="mean")
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
        metrics["F1@%f" % t] = f1.sum(axis=0)
    
    return


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

# TODO: render some sampled latent point clouds
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


def plot_heatmap(model, results_dir):
    '''
    Plot embeddings pairwise distance heatmap.
    '''
    embed = model.quantize.embed.data
    pairwise_dist = F.pdist(embed.t(), p=2)
    N = embed.t().shape[0]
    pairwise_distances = torch.zeros((N, N-1))
    temp = torch.triu_indices(N, N-1)
    pairwise_distances[temp[0], temp[1]] = pairwise_dist
    pairwise_distances = pairwise_distances.cpu().numpy()

    plt.figure(1)
    sns.set(rc={'figure.figsize':(4, 4)})
    mask = np.tril(np.ones_like(pairwise_distances, dtype=bool))
    sns.heatmap(pairwise_distances, cmap="BuGn_r", mask=mask, xticklabels=32, yticklabels=32)
    
    save_file_fig = os.path.join(results_dir, 'heatmap.png')
    plt.savefig(save_file_fig, bbox_inches="tight")
    save_file_embed = os.path.join(results_dir, 'embed.csv')
    np.savetxt(save_file_embed, embed.cpu().numpy(), delimiter = ',')


def plot_embeddings(model, results_dir):
    plt.figure(2)
    proj = umap.UMAP(n_neighbors=5,
                 min_dist=0.1,
                 metric='euclidean').fit_transform(model.quantize.embed.data.cpu().numpy())
    plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
    save_file = os.path.join(results_dir, 'view_embeds.png')
    plt.savefig(save_file, bbox_inches="tight")


# TODO: plot latent representations of input images
# See plot_latent in https://avandekleut.github.io/vae/
def plot_latents(images):
    pass



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
    print("Point clouds and plots will be saved in: " + results_dir)
    print("Metrics will be written to: " + metrics_file)

    model_name = config.model
    splits_path = os.path.join(orig_dir, config.splits_path)  # Load splits from original path
    data_dir = os.path.join('/home', config.data_dir)
    test_loader = build_data_loader(
        data_dir=data_dir,
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
    f1_01 = {syn_id: 0 for syn_id in syn_ids}
    f1_03 = {syn_id: 0 for syn_id in syn_ids}
    f1_05 = {syn_id: 0 for syn_id in syn_ids}

    print("Starting evaluation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, _, ptclds_gt, normals, RT, K, id_strs = test_loader.postprocess(batch)
            for syn_id in syn_ids:
                num_instances[syn_id] += len(images)
            
            if model_name == 'vqvae':
                ptclds_pred, l_vq, encoding_inds, perplexity = model(images, RT)
                ptclds_pred_sampled = model.latent_sample(config.bs)
            else:
                ptclds_pred = model(images, RT)
            cur_metrics = compute_metrics(ptclds_pred, ptclds_gt)

            for i, syn_id in enumerate(syn_ids):
                chamfer[syn_id] += cur_metrics['Chamfer'].item()
                f1_01[syn_id] += cur_metrics['F1@%f' % 0.1].item()
                f1_03[syn_id] += cur_metrics['F1@%f' % 0.3].item()
                f1_05[syn_id] += cur_metrics['F1@%f' % 0.5].item()

            if batch_idx % config.save_freq == 0:
                for i, sid in enumerate(syn_ids):
                    utils.save_point_clouds(id_strs[i] + str(batch_idx), ptclds_pred, ptclds_gt, results_dir)
                    if model_name == 'vqvae':
                        utils.save_sampled_ptclds(id_strs[i] + str(batch_idx), ptclds_pred_sampled, results_dir)
                        breakpoint()
    
        col_headers = [class_names[syn_id] for syn_id in syn_ids]
        chamfer_final = {class_names[syn_id] : cham_dist/num_instances[syn_id] for syn_id, cham_dist in chamfer.items()}
        f1_01_final = {class_names[syn_id] : f1_01/num_instances[syn_id] for syn_id, f1_01 in f1_01.items()}
        f1_03_final = {class_names[syn_id] : f1_03/num_instances[syn_id] for syn_id, f1_03 in f1_03.items()}
        f1_05_final = {class_names[syn_id] : f1_05/num_instances[syn_id] for syn_id, f1_05 in f1_05.items()}
        metrics = [chamfer_final, f1_01_final, f1_03_final, f1_05_final] 
        with open(metrics_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=col_headers)
            writer.writeheader()
            writer.writerows(metrics)

    print(f"Saved evaluation metrics to {metrics_file}")
    plot_heatmap(model, results_dir)
    plot_embeddings(model, results_dir)
    print(f"Saved plots to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-E', '--encoder_path', default=None, help='(Relative or absolute) path of the encoder checkpoint')
    parser.add_argument('-Q', '--quantize_path', default=None, help='(Relative or absolute) path of the quantization checkpoint')
    parser.add_argument('-D', '--decoder_path', default=None, help='(Relative or absolute) path of the decoder checkpoint')
    args = parser.parse_args()

    # Use decoder checkpoint directory as run directory
    checkpoints_dir = os.path.dirname(os.path.abspath('./outputs/2021-05-29/17-30-11/checkpoints/vqvae_epoch99_step51800.pth')) # /path/to/run_dir/checkpoints/checkpoint.pth
    run_dir = os.path.dirname(checkpoints_dir)

    config_dir = os.path.join(run_dir, '.hydra')
    initialize_config_dir(config_dir=config_dir, job_name='evaluate')
    overrides = [f"+{arg}={getattr(args, arg)}" for arg in vars(args) if getattr(args, arg)]
    overrides.append(f"+run_dir={run_dir}")
    config = compose(config_name='config', overrides=overrides)
    print(OmegaConf.to_yaml(config))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    evaluate(config)

