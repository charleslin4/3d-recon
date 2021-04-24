import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader, random_split
from in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder
from autoencoder import AutoEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--data_dir')

    top_in_dir = 'gs://3d-recon/data/shape_net_core_uniform_samples_2048/' #  './data/shape_net_core_uniform_samples_2048/'
    class_name = 'chair'
    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = os.path.join(top_in_dir, syn_id)
    dataset = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # TODO use config parameters
    ae = AutoEncoder(2048, 3, 128)
    wandb_logger = WandbLogger(name='test', project='3d-recon')
    trainer = pl.Trainer(gpus=args.gpus, logger=wandb_logger)
    trainer.fit(ae, DataLoader(train_dataset, batch_size=50, num_workers=2), DataLoader(val_dataset, batch_size=50, num_workers=2))
