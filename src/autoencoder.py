import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch3d.loss import chamfer
import pytorch3d.ops as ops 
import wandb
import torchvision.models as models
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.utils import ico_sphere

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True, progress=True)

    def forward(self, x):
        out = self.resnet50(x)

        return out


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.fc3_points = nn.Sequential(
            nn.Linear(512, 3),
            nn.Tanh()
        )

        self.fc3_textures = nn.Sequential(
            nn.Linear(512, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        src_mesh = ico_sphere(4, device=None)
        points = sample_points_from_meshes(src_mesh, 10000)
        out = ops.vert_align(feats=x, verts=points)
        out = self.fc1(out)
        out = self.fc2(out)
        out_points = self.fc3_points(out)
        out_textures = self.fc3_textures(out)
        
        return out_points, out_textures


class AutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

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

