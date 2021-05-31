import os
import tempfile
import random
import numpy as np
from datetime import datetime
import logging

import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from datautils.dataloader import build_data_loader
import utils

logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(config):

    run_dir = os.getcwd()
    orig_dir = hydra.utils.get_original_cwd()

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')  # Save checkpoints to run directory
    data_dir = hydra.utils.to_absolute_path(config.data_dir)
    splits_path = hydra.utils.to_absolute_path(config.splits_path)  # Load splits from original path
    deprocess_transform = utils.imagenet_deprocess()

    model_name = config.model
    ae = utils.load_model(model_name, config)
    ae.to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=config.lr)

    if hasattr(config, 'decoder'):
        model_name += f'_{config.decoder}{config.num_layers}'
    if not config.debug:
        date, time = run_dir.split('/')[-2:]
        timestamp = f"{date}.{time}"
        run_name = f'{model_name}_{timestamp}'
        wandb_dir = tempfile.mkdtemp()
        wandb.init(name=run_name, config=config, project='3d-recon', entity='3drecon2', dir=wandb_dir)

    global_iter = 0
    for epoch in range(config.epochs):

        train_loader = build_data_loader(
            data_dir=data_dir,
            split_name='train',
            splits_file=splits_path,
            batch_size=config.bs,
            num_workers=4,
            multigpu=False,
            shuffle=True,
            num_samples=None
        )

        for batch_idx, batch in enumerate(train_loader):
            images, meshes, ptclds_gt, normals, RT, K = train_loader.postprocess(batch)  # meshes, normals = None
            batch_size = images.shape[0]
            opt.zero_grad()
            ae.train()

            loss = 0.0
            info_dict = {'step': global_iter}

            if model_name == 'vqvae':
                ptclds_pred, l_vq, encoding_inds, perplexity = ae(images, RT)
                info_dict['loss/vq'] = l_vq.item()
                info_dict['ppl/train'] = perplexity.item()
                info_dict['encodings'] = wandb.Histogram(encoding_inds.flatten().cpu().numpy(), num_bins=256)
                loss += config.commitment_cost * l_vq
            elif model_name == 'vae':
                ptclds_pred, mu, logvar = ae(images, RT)
                l_kl = (-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)).mean()
                info_dict['loss/kl'] = l_kl.item()
                loss += config.beta * l_kl
            else:
                ptclds_pred = ae(images, RT)

            l_rec, _ = chamfer_distance(ptclds_gt, ptclds_pred)
            info_dict['loss/rec'] = l_rec.item()
            loss += l_rec

            loss.backward()
            info_dict['loss/train'] = loss.item()

            grads_encoder = nn.utils.clip_grad_norm_(ae.encoder.parameters(), 50)
            grads_decoder = nn.utils.clip_grad_norm_(ae.decoder.parameters(), 50)
            info_dict['grads/encoder'] = grads_encoder.item()
            info_dict['grads/decoder'] = grads_decoder.item()

            if global_iter % config.save_freq == 0:
                info_dict.update({
                    'image': wandb.Image(deprocess_transform(images)[0].permute(1, 2, 0).cpu().numpy()),
                    'pt_cloud/pred': wandb.Object3D(ptclds_pred[0].detach().cpu().numpy()),
                    'pt_cloud/gt': wandb.Object3D(ptclds_gt[0].detach().cpu().numpy())
                })

            logger.info("Epoch {}\tTrain step {}\tLoss: {:.2f}".format(epoch, batch_idx, loss.item()))
            if not config.debug:
                wandb.log(info_dict)

            opt.step()

            if global_iter % config.save_freq == 0 and not config.debug:
                logger.info(f'Saving the latest model (epoch {epoch}, global_iter {global_iter})')
                utils.save_checkpoint_model(ae, model_name, epoch, checkpoint_dir, global_iter)
 
            global_iter += 1

        val_loader = build_data_loader(
            data_dir=data_dir,
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
            info_dict = {
                'step': global_iter,
                'loss/val': 0.0
            }
            if model_name == 'vqvae':
                info_dict['ppl/val'] = 0.0

            for batch_idx, batch in enumerate(val_loader):
                images, meshes, ptclds_gt, normals, RT, K = val_loader.postprocess(batch)  # meshes, normals = None
                batch_size = images.shape[0]

                loss = 0.0
                if model_name == 'vqvae':
                    ptclds_pred, l_vq, encoding_inds, perplexity = ae(images, RT)
                    loss += config.commitment_cost * l_vq.item()
                elif model_name == 'vae':
                    ptclds_pred, mu, logvar = ae(images, RT)
                    l_kl = (-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)).mean()
                    loss += config.beta * l_kl
                else:
                    ptclds_pred = ae(images, RT)

                l_rec, _ = chamfer_distance(ptclds_gt, ptclds_pred)
                loss += l_rec.item()

                info_dict['loss/val'] += loss * batch_size
                if model_name == 'vqvae':
                    info_dict['ppl/val'] += perplexity.item() * batch_size

            info_dict['loss/val'] /= len(val_loader.dataset)
            if model_name == 'vqvae':
                info_dict['ppl/val'] /= len(val_loader.dataset)

            logger.info("Epoch {} Validation\tLoss: {:.2f}".format(epoch, info_dict['loss/val']))
            if not config.debug:
                wandb.log(info_dict)

    if not config.debug:
        logger.info(f'Saving the latest model (epoch {epoch}, global_iter {global_iter})')
        utils.save_checkpoint_model(ae, model_name, epoch, checkpoint_dir, global_iter)


@hydra.main(config_path='config', config_name='config')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train(config)


if __name__ == "__main__":
    main()

