import os
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from pointalign import PointAlign
from dataloader import build_data_loader
import utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(config):

    run_dir = os.getcwd()
    orig_dir = hydra.utils.get_original_cwd()

    # NOTE This is all sort of janky, since we have a shared data directory on the Google Cloud VM.
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    model_name = f'{config.model}_vq_{timestamp}' if config.vq else f'{config.model}_{timestamp}'
    checkpoint_dir = os.path.join(run_dir, config.checkpoint_dir)  # Save checkpoints to run directory

    points_path = os.path.join(orig_dir, config.points_path)  # Load points from original path
    points = utils.load_point_sphere(points_path)
    splits_path = os.path.join(orig_dir, config.splits_path)  # Load splits from original path
    deprocess_transform = utils.imagenet_deprocess()

    ae = PointAlign(points).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=5e-4)

    wandb.init(name=model_name, project='3d-recon', entity='3drecon2')

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

        for batch_idx, batch in enumerate(train_loader):
            images, _, ptclds_gt, normals, RT, K = train_loader.postprocess(batch)
            batch_size = images.shape[0]
            opt.zero_grad()
            ae.train()

            loss = 0.0
            info_dict = {}

            ptclds_pred = ae(images, RT)
            l_rec, _ = chamfer_distance(ptclds_gt, ptclds_pred)
            l_rec.backward()
            info_dict['l_rec'] = l_rec.item()
            loss += l_rec.item()

            if config.vq:
                # TODO Add additional VQ-VAE loss terms here
                pass

            info_dict['loss'] = loss

            grads = nn.utils.clip_grad_norm_(ae.parameters(), 50)
            info_dict['grads'] = grads
            opt.step()

            print("Epoch {}\tTrain step {}\tLoss: {:.2f}".format(epoch, batch_idx, loss))
            wandb.log(info_dict)

            if batch_idx % 250 == 0:
                wandb.log({
                    'image': wandb.Image(deprocess_transform(images)[0].permute(1, 2, 0).cpu().numpy()),
                    'pt_cloud/pred': wandb.Object3D(ptclds_pred[0].detach().cpu().numpy()),
                    'pt_cloud/gt': wandb.Object3D(ptclds_gt[0].detach().cpu().numpy())
                })

            if global_iter % config.save_freq == 0:
                print(f'Saving the latest model (epoch {epoch}, global_iter {global_iter})')
                utils.save_checkpoint_model(ae, model_name, epoch, loss, checkpoint_dir, global_iter)
            
            global_iter += 1
            
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

