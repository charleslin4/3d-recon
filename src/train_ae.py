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
from models.vqvae import VQVAE, PointTransformer
from datautils.dataloader import build_data_loader
import utils

torch.multiprocessing.set_sharing_strategy('file_system')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(config):

    run_dir = os.getcwd()
    orig_dir = hydra.utils.get_original_cwd()

    # NOTE This is all sort of janky, since we have a shared data directory on the Google Cloud VM.
    model_name = config.model
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')  # Save checkpoints to run directory

    points_path = hydra.utils.to_absolute_path(config.points_path)  # Load points from original path
    points = utils.load_point_sphere(points_path)
    splits_path =  hydra.utils.to_absolute_path(config.splits_path)  # Load splits from original path
    deprocess_transform = utils.imagenet_deprocess()

    # TODO Handle cases where config does not contain kwargs
    if model_name == 'pointalign':
        ae = PointAlign(
            points,
            hidden_dim=config.hidden_dim
        )

    elif model_name == 'pointalignsmall':
        ae = PointAlignSmall(
            points,
            hidden_dim=config.hidden_dim
        )

    elif model_name == 'vqvae':
        ae = VQVAE(
            points,
            hidden_dim=config.hidden_dim,
            num_embed=config.num_embed,
            embed_dim=config.embed_dim
        )

    elif model_name == 'transformer':
        ae = PointTransformer(
            points,
            num_embed=config.num_embed,
            embed_dim=config.embed_dim,
            nhead=config.nhead,
            num_layers=config.num_layers
        )

        if config.encoder_path:
            encoder_path = hydra.utils.to_absolute_path(config.encoder_path)
            ae.encoder.load_state_dict(torch.load(encoder_path)['model_state_dict'])

        if config.quantize_path:
            quantize_path = hydra.utils.to_absolute_path(config.quantize_path)
            ae.quantize.load_state_dict(torch.load(quantize_path)['model_state_dict'])

    else:
        raise Exception("Model options are `pointalign`, `pointalignsmall`, `vqvae`, or `transformer`.")

    ae.to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=config.lr)

    date, time = run_dir.split('/')[-2:]
    timestamp = f"{date}.{time}"
    if not config.debug:
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

            if model_name == 'vqvae':
                ptclds_pred, l_vq, encoding_inds, perplexity = ae(images, RT)
                info_dict['loss/vq'] = l_vq.item()
                info_dict['ppl/train'] = perplexity.item()
                info_dict['encodings'] = wandb.Histogram(encoding_inds.flatten().cpu().numpy(), num_bins=256)
                loss += config.commitment_cost * l_vq
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

            if global_iter % 250 == 0:
                info_dict.update({
                    'image': wandb.Image(deprocess_transform(images)[0].permute(1, 2, 0).cpu().numpy()),
                    'pt_cloud/pred': wandb.Object3D(ptclds_pred[0].detach().cpu().numpy()),
                    'pt_cloud/gt': wandb.Object3D(ptclds_gt[0].detach().cpu().numpy())
                })

            print("Epoch {}\tTrain step {}\tLoss: {:.2f}".format(epoch, batch_idx, loss.item()))
            if not config.debug:
                wandb.log(info_dict)

            opt.step()

            if global_iter % config.save_freq == 0 and not config.debug:
                print(f'Saving the latest model (epoch {epoch}, global_iter {global_iter})')
                if model_name == 'VQVAE':
                    utils.save_checkpoint_model(ae.encoder, model_name+'_Encoder', epoch, checkpoint_dir, global_iter)
                    utils.save_checkpoint_model(ae.quantize, model_name+'_Quantize', epoch, checkpoint_dir, global_iter)
                    utils.save_checkpoint_model(ae.decoder, model_name+'_Decoder', epoch, checkpoint_dir, global_iter)
                else:
                    utils.save_checkpoint_model(ae, model_name, epoch, checkpoint_dir, global_iter)
            
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
            print("Epoch {} Validation\tLoss: {:.2f}".format(epoch, info_dict['loss/val']))
            if not config.debug:
                wandb.log(info_dict)

    if not config.debug:
        print(f'Saving the latest model (epoch {epoch}, global_iter {global_iter})')
        if model_name == 'vqvae':
            utils.save_checkpoint_model(ae.encoder, model_name+'_Encoder', epoch, checkpoint_dir, global_iter)
            utils.save_checkpoint_model(ae.quantize, model_name+'_Quantize', epoch, checkpoint_dir, global_iter)
            utils.save_checkpoint_model(ae.decoder, model_name+'_Decoder', epoch, checkpoint_dir, global_iter)
        else:
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

