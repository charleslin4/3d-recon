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
from utils import save_checkpoint_model, rotate_verts, imagenet_deprocess

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(args):

    ae = PointAlign().to(DEVICE)
    opt = torch.optim.SGD(ae.parameters(), lr=1e-2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)

    deprocess_transform = imagenet_deprocess()
    # Rotation matrix to transform the ground truth point clouds to the same views as the input images
    rot = torch.tensor([[0, -1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], device=DEVICE)

    global_iter = 0
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
        breakpoint()

        for batch_idx, batch in enumerate(train_loader):
            images, _, ptclds_gt, normals, RT, K = train_loader.postprocess(batch)
            batch_size = images.shape[0]
            opt.zero_grad()
            ae.train()

            ptclds_pred, textures_pred = ae(images)
            ptclds_gt_cam = rotate_verts(rot.to(RT).unsqueeze(0).repeat(batch_size, 1, 1), ptclds_gt)
            loss, _ = chamfer_distance(ptclds_gt_cam, ptclds_pred)
            loss.backward()

            grads = nn.utils.clip_grad_norm_(ae.parameters(), 50)
            opt.step()

            print("Epoch {}\tTrain step {}\tLoss: {:.2f}".format(epoch, batch_idx, loss))
            wandb.log({
                'loss': loss,
                'grads': grads
            })

            if batch_idx % 25 == 0:
                wandb.log({
                    'image': wandb.Image(deprocess_transform(images)[0].permute(1, 2, 0).cpu().numpy()),
                    'pt_cloud/pred': wandb.Object3D(ptclds_pred[0].detach().cpu().numpy()),
                    'pt_cloud/gt': wandb.Object3D(ptclds_gt_cam[0].detach().cpu().numpy())
                })

            if global_iter % args.save_freq == 0:
                print('Saving the latest model (epoch %d, global_iter %d)' % (epoch, global_iter))
                save_checkpoint_model(ae, args.model_name, epoch, loss, args.checkpoint_dir, global_iter)
            
            global_iter += 1

        scheduler.step()


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

