import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch3d.loss import chamfer
from general_utils import plot_3d_point_cloud
import wandb
import matplotlib.pyplot as plt

class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, points_per_cloud):
        super(Encoder, self).__init__()
        
        self.conv1_bn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.conv2_bn = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv3_bn = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv4_bn = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.conv5_bn = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv1_bn(x)
        out = self.conv2_bn(out)
        out = self.conv3_bn(out)
        out = self.conv4_bn(out)
        out = self.conv5_bn(out)
        out, _ = torch.max(out, 2)

        return out


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.fc1_bn = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc2_bn = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, out_channels)
        )

    def forward(self, x):
        out = self.fc1_bn(x)
        out = self.fc2_bn(out)
        out = self.fc3(out)
        
        return out


class AutoEncoder(pl.LightningModule):

    def __init__(self, points_per_cloud, in_channels, bneck_size, loss='chamfer'):
        super().__init__()

        self.points_per_cloud = points_per_cloud
        self.in_channels = in_channels
        self.bneck_size = bneck_size
        self.loss = loss.lower()

        self.encoder = Encoder(in_channels, bneck_size, points_per_cloud)
        self.decoder = Decoder(bneck_size, in_channels*points_per_cloud)

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x = batch
        x = x.permute(0, 2, 1)
        z = self.encoder(x)
        x_hat = self.decoder(z).view(-1, self.in_channels, self.points_per_cloud)

        if self.loss == 'chamfer':
            loss = chamfer.chamfer_distance(x_hat, x)[0]
        elif loss == 'emd':
            # TODO
            print("Not yet implemented.")
            import pdb; pdb.set_trace()
        else:
            print("Please choose either 'chamfer' or 'emd' for the loss.")
            import pdb; pdb.set_trace()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.permute(0, 2, 1)
        z = self.encoder(x)
        x_hat = self.decoder(z).view(-1, self.in_channels, self.points_per_cloud)

        if self.loss == 'chamfer':
            loss = chamfer.chamfer_distance(x_hat, x)[0]
        elif loss == 'emd':
            # TODO
            print("Not yet implemented.")
            import pdb; pdb.set_trace()
        else:
            print("Please choose either 'chamfer' or 'emd' for the loss.")
            import pdb; pdb.set_trace()

        self.log('val_loss', loss)

        if batch_idx % 10 == 0:
            recon = x_hat[0].cpu().numpy().T
            wandb.log({'eval': wandb.Object3D(recon)})
            gt = x[0].cpu().numpy().T
            wandb.log({'ground_truth': wandb.Object3D(gt)})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=0.001)
        return optimizer

