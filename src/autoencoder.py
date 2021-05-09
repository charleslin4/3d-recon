import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from pytorch3d.ops import sample_points_from_meshes, vert_align
from pytorch3d.utils import ico_sphere

import numpy as np

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
        point_diffs = self.fc3_points(out)
        out_textures = self.fc3_textures(out)

        out_points = points - point_diffs
        return out_points, out_textures  # At the moment, returning a Pointclouds object will create an infinite loop


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, images):
        z = self.encoder(images)
        points, textures = self.decoder(z)

        return points, textures

