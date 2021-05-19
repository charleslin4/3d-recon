import os
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from models.pointalign import PointAlign, PointAlignSmall
from models.vqvae import VQVAE
from datautils.dataloader import build_data_loader
import utils

torch.multiprocessing.set_sharing_strategy('file_system')  # Potential fix to 'received 0 items of ancdata' error
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
commitment_cost = 0.25 # beta in Eq 2 of VQVAE 2 paper

def train(config):

    run_dir = os.getcwd()
    orig_dir = hydra.utils.get_original_cwd()

    # NOTE This is all sort of janky, since we have a shared data directory on the Google Cloud VM.
    model_name = config.model
    if config.vq:
        model_name += '_vq'
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')  # Save checkpoints to run directory

    points_path = os.path.join(orig_dir, config.points_path)  # Load points from original path
    points = utils.load_point_sphere(points_path)
    splits_path = os.path.join(orig_dir, config.splits_path)  # Load splits from original path
    deprocess_transform = utils.imagenet_deprocess()

    if config.model == 'PointAlign':
        ae = PointAlign(points)
    elif config.vq:
        ae = VQVAE(points)
    else:
        ae = PointAlignSmall(points)
    ae.to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=config.lr)

    date, time = run_dir.split('/')[-2:]
    timestamp = f"{date}.{time}"
    wandb.init(name=f'{model_name}_{timestamp}', config=config, project='3d-recon', entity='3drecon2', dir=orig_dir)
    wandb.watch(ae)

    global_iter = 0
    for epoch in range(config.epochs):

        train_loader = build_data_loader(
            data_dir=config.data_dir,
            split_name='train',
            splits_file=splits_path,
            batch_size=config.bs,
            num_workers=4,
            multigpu=False,
            shuffle=True,
            num_samples=None
        )

        ae.train()
        for batch_idx, batch in enumerate(train_loader):
            images, meshes, ptclds_gt, normals, RT, K = train_loader.postprocess(batch)  # meshes, normals = None
            batch_size = images.shape[0]
            opt.zero_grad()
            ae.train()

            loss = 0.0
            info_dict = {'step': global_iter}

            if config.vq:
                ptclds_pred, l_vq, perplexity = ae(images, RT)
                ptclds_pred.to(DEVICE)
                l_vq.to(DEVICE)
                perplexity.to(DEVICE)
                l_rec, _ = chamfer_distance(ptclds_gt, ptclds_pred)
                info_dict['loss/rec'] = l_rec.item()
                info_dict['loss/vq'] = l_vq.item()
                info_dict['perplexity'] = perplexity.item()
                vq_loss = l_rec + commitment_cost * l_vq
                vq_loss.backward()
                loss += vq_loss.item()
            else:
                ptclds_pred = ae(images, RT)
                l_rec, _ = chamfer_distance(ptclds_gt, ptclds_pred)
                l_rec.backward()
                info_dict['loss/rec'] = l_rec.item()
                loss += l_rec.item()

            info_dict['loss/train'] = loss

            grads = nn.utils.clip_grad_norm_(ae.parameters(), 50)
            info_dict['grads'] = grads.item()

            if global_iter % 250 == 0:
                info_dict.update({
                    'image': wandb.Image(deprocess_transform(images)[0].permute(1, 2, 0).cpu().numpy()),
                    'pt_cloud/pred': wandb.Object3D(ptclds_pred[0].detach().cpu().numpy()),
                    'pt_cloud/gt': wandb.Object3D(ptclds_gt[0].detach().cpu().numpy())
                })

            print("Epoch {}\tTrain step {}\tLoss: {:.2f}".format(epoch, batch_idx, loss))
            wandb.log(info_dict)

            opt.step()

            if global_iter % config.save_freq == 0:
                print(f'Saving the latest model (epoch {epoch}, global_iter {global_iter})')
                utils.save_checkpoint_model(ae, model_name, epoch, loss, checkpoint_dir, global_iter)
            
            global_iter += 1

        val_loader = build_data_loader(
            data_dir=config.data_dir,
            split_name='val',
            splits_file=splits_path,
            batch_size=config.bs,
            num_workers=4,
            multigpu=False,
            shuffle=True,
            num_samples=None
        )

        ae.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for batch_idx, batch in enumerate(val_loader):
                images, meshes, ptclds_gt, normals, RT, K = val_loader.postprocess(batch)  # meshes, normals = None
                batch_size = images.shape[0]

                loss = 0.0
                if config.vq:
                    ptclds_pred, l_vq = ae(images, RT)
                    l_rec, _ = chamfer_distance(ptclds_gt, ptclds_pred)
                    loss += l_rec.item() + commitment_cost * l_vq.item()
                else:
                    ptclds_pred = ae(images, RT)
                    l_rec, _ = chamfer_distance(ptclds_gt, ptclds_pred)
                    loss += l_rec.item()

                total_val_loss += loss * batch_size

            val_loss = total_val_loss / len(val_loader.dataset)
            print("Epoch {} Validation\tLoss: {:.2f}".format(epoch, val_loss))
            wandb.log({'step': global_iter, 'loss/val': val_loss})

        print(f'Saving the latest model (epoch {epoch}, global_iter {global_iter})')
        utils.save_checkpoint_model(ae, model_name, epoch, loss, checkpoint_dir, global_iter)


@hydra.main(config_path='config', config_name='train_ae')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train(config)


if __name__ == "__main__":
    main()

