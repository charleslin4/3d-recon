import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

import wandb
from pointalign import PointAlign
from dataloader import build_data_loader
from utils import save_checkpoint_model, rotate_verts


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(args):

    ae = PointAlign().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-4)
    wandb.watch(ae, log_freq=1)

    total_iters = 0
    for epoch in range(args.epochs):

        train_loader = build_data_loader(
            data_dir=args.data_dir,
            split_name='train',
            splits_file=args.splits_path,
            batch_size=args.bs,
            num_workers=4,
            multigpu=False,
            shuffle=True,
            num_samples=None
        )

        for batch_idx, batch in enumerate(train_loader):
            total_iters += args.bs
            
            images, _, ptclds_gt, normals, RT, K = train_loader.postprocess(batch)
            opt.zero_grad()
            ae.train()

            ptclds_pred, textures_pred = ae(images)
            ptclds_gt_cam = rotate_verts(RT, ptclds_gt)
            loss, _ = chamfer_distance(ptclds_gt, ptclds_pred)
            loss.backward()
            opt.step()

            print("Epoch {}\tTrain step {}\tLoss: {:.2f}".format(epoch, batch_idx, loss))
            wandb.log({
                'loss': loss,
                'pt_cloud/pred': wandb.Object3D(ptclds_pred[0].detach().cpu().numpy()),
                'pt_cloud/gt': wandb.Object3D(ptclds_gt_cam[0].detach().cpu().numpy())
            })

            if total_iters % args.save_freq == 0:
                print('Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_checkpoint_model(ae, args.model_name, epoch, loss, args.checkpoint_dir, total_iters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--data_dir', type=str, default='/home/data/ShapeNet/ShapeNetV1processed')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--model_name', type=str, default='baseline')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/checkpoints')
    parser.add_argument('--splits_path', type=str, default='./data/bench_splits.json')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wandb.init(project='3d-recon', entity='3drecon2')
    train(args)

