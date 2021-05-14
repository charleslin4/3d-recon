import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from pytorch3d.datasets.r2n2.utils import project_verts
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import vert_align

import numpy as np
import utils
from backbone import build_backbone

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Decoder(nn.Module):

    def __init__(self, points, img_feat_dim=2048, hidden_dim=512):
        super(Decoder, self).__init__()

        self.points = points.to(DEVICE)

        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + 3, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + 3, hidden_dim)

        self.point_offset = nn.Linear(hidden_dim + 3, 3)
        self.point_texture = nn.Linear(hidden_dim + 3, 3)


    def forward(self, x, P=None):
        point_spheres = self.points.repeat(x.shape[0], 1, 1)

        if P is not None:
            point_spheres = project_verts(point_spheres, P)

        device, dtype = point_spheres.device, point_spheres.dtype
        factor = torch.tensor([1, -1, 1], device=device, dtype=dtype).view(1, 1, 3)
        point_spheres = point_spheres * factor

        vert_align_feats = vert_align(x, point_spheres)

        vert_feats_nopos = F.relu(self.bottleneck(vert_align_feats))
        vert_feats = torch.cat([vert_feats_nopos, point_spheres], dim=-1)

        vert_feats_nopos = F.relu(self.fc1(vert_feats))
        vert_feats = torch.cat([vert_feats_nopos, point_spheres], dim=-1)

        vert_feats_nopos = F.relu(self.fc2(vert_feats))
        vert_feats = torch.cat([vert_feats_nopos, point_spheres], dim=-1)

        vert_feats_nopos = F.relu(self.fc3(vert_feats))
        vert_feats = torch.cat([vert_feats_nopos, point_spheres], dim=-1)

        point_offsets = torch.tanh(self.point_offset(vert_feats))

        out_points = point_spheres + point_offsets
        return out_points


class PointAlign(nn.Module):

    def __init__(self, points):
        super().__init__()

        self.encoder, feat_dims = build_backbone('resnet50', pretrained=True)
        self.decoder = Decoder(points, feat_dims[-1], 512)

    def forward(self, images, P=None):
        z = self.encoder(images)[-1]
        points = self.decoder(z, P)

        return points

