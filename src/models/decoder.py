import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.datasets.r2n2.utils import project_verts
from pytorch3d.ops import vert_align


class FCDecoder(nn.Module):

    def __init__(self, points=None, input_dim=512, hidden_dim=128, num_layers=1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if points is None:
            points = torch.randn((1, 10000, 3))
        self.register_buffer('points', points)

        self.bottleneck = nn.Linear(input_dim, hidden_dim)
        for layer in range(self.num_layers):
            setattr(self, f"fc{layer+1}", nn.Linear(hidden_dim + 3, hidden_dim))
        self.point_offset = nn.Linear(hidden_dim + 3, 3)


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

        for layer in range(self.num_layers):
            vert_feats_nopos = F.relu(getattr(self, f"fc{layer+1}")(vert_feats))
            vert_feats = torch.cat([vert_feats_nopos, point_spheres], dim=-1)

        point_offsets = torch.tanh(self.point_offset(vert_feats))

        out_points = point_spheres + point_offsets
        return out_points

