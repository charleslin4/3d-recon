import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import pytorch_lightning as pl

from pytorch3d.datasets import BlenderCamera
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes, vert_align
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PointLights,
    AlphaCompositor,
    NormWeightedCompositor
)

import numpy as np
import wandb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample_points_from_sphere(clouds, points_per_cloud=10000):
    src_mesh = ico_sphere(2, device=DEVICE)
    points = sample_points_from_meshes(src_mesh, clouds * points_per_cloud)
    points = points.reshape(clouds, points_per_cloud, -1)
    return points


def resnet_backbone(
    backbone_name,
    pretrained,
    norm_layer=torchvision.ops.misc.FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None
):

    fpn_backbone = resnet_fpn_backbone(backbone_name, pretrained, norm_layer, trainable_layers, returned_layers, extra_blocks)
    modules = list(fpn_backbone.children())[:-1]
    backbone = nn.Sequential(*modules)
    return backbone


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.model = resnet_backbone('resnet50', pretrained=True, trainable_layers=3)

    def forward(self, x):
        out = self.model(x)['3']  # get the output of the last residual block

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
        points = sample_points_from_sphere(x.shape[0], 10000)
        out = vert_align(feats=x, verts=points)
        out = self.fc1(out)
        out = self.fc2(out)
        out_points = self.fc3_points(out)
        out_textures = self.fc3_textures(out)
        
        point_clouds = Pointclouds(out_points, features=out_textures)
        return point_clouds


class AutoEncoder(pl.LightningModule):

    def __init__(self, raster_settings=PointsRasterizationSettings(image_size=137)):
        super().__init__()

        self.raster_settings = raster_settings

        self.encoder = Encoder()
        self.decoder = Decoder()

    def render_points(self, point_clouds, R, T, K):
        camera = FoVOrthographicCameras(device=self.device, R=R, T=T, K=K)
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=camera, 
                raster_settings=self.raster_settings
            ),
            compositor=AlphaCompositor()
        )

        # TODO 
        images = renderer(point_clouds, cameras=camera, lights=PointLights())  # (N, H, W, C)
        silhouettes = 1. - (1 - images).prod(dim=-1)
        images = images.permute(0, 3, 1, 2)  # (N, C, H, W)
        return images, silhouettes


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        try:
            images = batch['images']
            batch_size, n_views = images.shape[0], images.shape[1]
            idx = np.asarray([np.random.choice(n_views, 2, replace=False) for i in range(batch_size)])
            input_idx, support_idx = idx[:, 0], idx[:, 1]

            input_images = images[range(batch_size), input_idx]     # (N, H, W, C)
            support_images = images[range(batch_size), support_idx] # (N, H, W, C)
            input_masks = 1. - (1 - input_images).prod(dim=-1)
            support_masks = 1. - (1 - support_images).prod(dim=-1)
            input_images = input_images.permute(0, 3, 1, 2)      # (N, C, H, W)
            support_images = support_images.permute(0, 3, 1, 2)  # (N, C, H, W)

            z = self.encoder(input_images)
            point_clouds = self.decoder(z)

            R, T, K = batch['R'], batch['T'], batch['K']
            out_images, out_masks = self.render_points(
                point_clouds,
                R[range(batch_size), input_idx],
                T[range(batch_size), input_idx],
                K[range(batch_size), input_idx]
            )

            out_support_images, out_support_masks = self.render_points(
                point_clouds,
                R[range(batch_size), support_idx],
                T[range(batch_size), support_idx],
                K[range(batch_size), support_idx]
            )

            l_base_mask = (
                1. - (input_masks * out_masks).sum(dim=(1,2)) / ((input_masks + out_masks - input_masks * out_masks).sum(dim=(1,2)))
            ).mean()
            l_base_l1 = F.l1_loss(input_images, out_images)
            l_base = l_base_mask + l_base_l1

            l_support_mask = (
                1. - (support_masks * out_support_masks).sum(dim=(1,2)) / ((support_masks + out_support_masks - support_masks * out_support_masks).sum(dim=(1,2)))
            ).mean()
            l_support_l1 = F.l1_loss(support_images, out_support_images)
            l_support = l_support_mask + l_support_l1

            loss = l_base + l_support

            self.log('loss/train', loss)
            self.log('loss/base_mask', l_base_mask)
            self.log('loss/base_l1', l_base_l1)
            self.log('loss/support', l_support)
            self.log('loss/support_mask', l_support_mask)
            self.log('loss/support_l1', l_support_l1)
        
        except Exception as e:
            loss = None
            print(e)

        return loss


    def validation_step(self, batch, batch_idx):
        '''
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
        '''
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


