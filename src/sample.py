import os
import random
import argparse
from datetime import datetime
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
from hydra.experimental import compose, initialize_config_dir

import utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample(config: DictConfig, num_samples):
    orig_dir = os.getcwd()
    results_dir = os.path.join(config.run_dir, 'samples')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d.%H-%M-%S")

    model_name = config.model
    assert model_name in ['vae', 'vqvae'], 'Please choose a model which can be sampled'
    model = utils.load_model(model_name, config)
    model.to(DEVICE)
    model.eval()

    ptclds, inds, all_latents = [], [], []

    with torch.no_grad():
        for sample_idx in range(0, num_samples, config.bs):
            bs = num_samples - sample_idx if num_samples - sample_idx < config.bs else config.bs

            if model_name == 'vae':
                mu = torch.zeros((bs, config.latent_dim), device=DEVICE)
                logvar = torch.zeros((bs, config.latent_dim), device=DEVICE)
                latents = model.latent_sample(mu, logvar)
                all_latents.append(latents)
                sampled_ptclds = model.decode(latents)
                ptclds.append(sampled_ptclds)

            elif model_name == 'vqvae':
                sampled_inds, latents = model.latent_sample(bs)
                sampled_ptclds = model.decode(latents)
                ptclds.append(sampled_ptclds)
                inds.append(sampled_inds)

    ptclds_path = os.path.join(results_dir, f'ptclds_{timestamp}.pt')
    torch.save(torch.cat(ptclds, dim=0), ptclds_path)
    print(f'Saved point clouds to {ptclds_path}')
    if inds:
        inds_path = os.path.join(results_dir, f'inds_{timestamp}.pt')
        torch.save(torch.cat(inds, dim=0), inds_path)
        print(f'Saved indices to {inds_path}')
    if all_latents:
        latents_path = os.path.join(results_dir, f'vae_latents_{timestamp}.pt')
        torch.save(torch.cat(all_latents, dim=0), latents_path)
        print(f'Saved VAE latents to {latents_path}')

if __name__ == "__main__":
    #path = './outputs/2021-05-29/17-30-11/checkpoints/vqvae_epoch99_step51800.pth'
    path = 'outputs/2021-05-31/17-28-11/checkpoints/vae_epoch99_step51800.pth'

    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--checkpoint_path', default=path, help='(Relative or absolute) path of the model checkpoint')
    parser.add_argument('-N', '--num_samples', type=int, default=1000)
    args = parser.parse_args()

    # Use decoder checkpoint directory as run directory
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

    sample(config, args.num_samples)

