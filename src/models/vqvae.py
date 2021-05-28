import torch
from torch import nn
from torch.nn import functional as F
from models.backbone import build_backbone
from models.pointalign import SmallDecoder
from pytorch3d.datasets.r2n2.utils import project_verts
from pytorch3d.ops import vert_align


class VQVAE_Encoder(nn.Module):
    def __init__(self, embed_dim=144):
        super().__init__()

        self.embed_dim = embed_dim

        self.backbone, feat_dims = build_backbone('resnet18', pretrained=True)  # (64, 128, 256, 512)
        self.bottleneck = torch.nn.Conv2d(in_channels=feat_dims[-1], out_channels=embed_dim, kernel_size=1)

    def forward(self, images):
        img_feats = F.relu(self.backbone(images)[-1])  # (64, 512, 5, 5)
        img_feats = self.bottleneck(img_feats)

        return img_feats


class VQVAE_Decoder(nn.Module):
    def __init__(self, points=None, embed_dim=144, nhead=48, num_layers=6):
        super().__init__()

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers

        if points is None:
            points = torch.randn((1, 10000, 3))
        self.register_buffer('points', points)

        decoder_layer = nn.TransformerDecoderLayer(embed_dim, nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.point_offset = nn.Linear(3 + 3, 3)

    def forward(self, memory, P=None):
        point_spheres = self.points.repeat(memory.shape[0], 1, 1)
        if P is not None:
            point_spheres = project_verts(point_spheres, P)

        point_spheres_ = point_spheres.reshape(memory.shape[0], -1, self.embed_dim).permute(1, 0, 2)  # (T, N, E)
        memory_ = memory.permute(1, 0, 2)  # (S, N, E)

        transformer_out = self.transformer_decoder(point_spheres_, memory_)
        vert_feats_nopos = transformer_out.reshape(-1, memory.shape[0], 3).permute(1, 0, 2)  # (N, 10000, 3)
        vert_feats = torch.cat([vert_feats_nopos, point_spheres], dim=-1)

        point_offsets = torch.tanh(self.point_offset(vert_feats))

        out_points = point_spheres + point_offsets
        return out_points


class VectorQuantizer(nn.Module):
    '''
    Represents the VQ-VAE layer with exponential moving averages to update embedding vectors.
    This makes embedding updates independent of the choice of optimizer, encoder, decoder, and other
    parts of the architecture and accelerates training.
    Implements the algorithm in 'Generating Diverse High Fidelity Images with VQ-VAE2' by Razavi et al.
    https://arxiv.org/pdf/1906.00446
    
    Adapted from: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py,
                  https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py.

    Attributes:
        num_embed: int, the number of codebook vectors in the latent space
        embed_dim: int, the dimensionality of each codebook vector in the latent space
        commitment_cost: scalar that controls weighting of commitment loss term
                        (beta in Eq 2 of the VQ-VAE 2 paper)
        decay: float, decay for the moving averages (gamma in Eq 2 of the VQ-VAE 2 paper)
        eps: small float constant to avoid numerical instability
        embed: encoded latent representations for the input
        cluster_size: N_i^(t) in Eq 2 of the VQVAE2 paper
        dw: m^(t) in Eq 2 of the VQVAE2 paper
    '''
    
    def __init__(self, num_embed, embed_dim, decay=0.99, eps=1e-5):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.decay = decay
        self.eps = eps

        embed = torch.empty(embed_dim, num_embed)
        nn.init.xavier_uniform_(embed)
        embed.requires_grad = False
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embed))
        self.register_buffer("dw", embed.clone())
    

    def forward(self, x):
        '''
        Forward pass of the vector quantization layer.

        Inputs:
            x - Tensor of shape (N, H, W, embed_dim)
        
        Returns:
            quantized: Tensor containing the quantized version of the input
            loss: Tensor containing the loss to optimize
            encoding_inds: Tensor containing the discrete encoding indices (which codebook
                           vector each input element was mapped to).
            perplexity: Tensor containing the perplexity of the encodings (i.e. average
                        codebook usage).
        '''

        x_flat = x.reshape(-1, self.embed_dim) # (NHW, embed_dim)
        
        dists = (x_flat**2).sum(dim=1, keepdim=True) - 2 * (x_flat @ self.embed) + (self.embed**2).sum(dim=0, keepdim=True) #(NHW, num_embed)
        
        encoding_inds = dists.argmin(dim=-1) # (NHW,)
        encodings = F.one_hot(encoding_inds, self.num_embed).type(x_flat.dtype) # (NHW, num_embed)
        assert x.shape[-1:][0] == self.embed_dim
        encoding_inds = encoding_inds.view(*x.shape[:-1]) # (N, H, W)

        quantized = self.quantize(encoding_inds) # (N, H, W, embed_dim)
        
        # Use Exponential Moving Average to update embedding vectors instead of using codebook loss
        # (ie., 2nd loss term in Eq 2 of VQVAE2 paper)
        if self.training:
            with torch.no_grad():
                # N^(t)
                self.cluster_size = self.decay * self.cluster_size + (1-self.decay) * encodings.sum(dim=0)
                # m^(t) = self.decay * m^(t-1) + (1-self.decay) * \sum_{j}^{n^(t)} E(x)_{i,j}^(t)
                self.dw = self.decay * self.dw + (1 - self.decay) * (x_flat.T @ encodings)
            
                n = self.cluster_size.sum()
                cluster_size = n * (self.cluster_size + self.eps) / (n + self.eps * self.num_embed)
                # e^(t) = m^(t) / N^(t)
                self.embed = self.dw / cluster_size.unsqueeze(0)
        
        # Compute difference between quantized codes and inputs
        diff = ((quantized.detach() - x)**2).mean()
        
        quantized = x + (quantized - x).detach()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, diff, encoding_inds, perplexity
    

    def quantize(self, embed_ind):
        return F.embedding(embed_ind, self.embed.T)


class VQVAE(nn.Module):

    def __init__(self, points=None, hidden_dim=128, num_embed=256, embed_dim=144):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        self.encoder = VQVAE_Encoder(embed_dim)
        self.quantize = VectorQuantizer(num_embed, embed_dim=embed_dim)
        self.decoder = SmallDecoder(points, input_dim=embed_dim, hidden_dim=hidden_dim)


    def forward(self, images, P=None):
        img_feats = self.encoder(images)
        img_feats = img_feats.permute(0, 2, 3, 1)  # (N, H, W, C)

        quant, diff, encoding_inds, perplexity = self.quantize(img_feats)
        quant = quant.permute(0, 3, 1, 2)  # (N, C, H, W)
        l_vq = diff.unsqueeze(0)

        ptclds = self.decoder(quant, P)

        return ptclds, l_vq, encoding_inds, perplexity


class PointTransformer(nn.Module):
    def __init__(self, points=None, num_embed=256, embed_dim=144, nhead=48, num_layers=6):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers

        self.encoder = VQVAE_Encoder(embed_dim)
        self.quantize = VectorQuantizer(num_embed, embed_dim)
        self.decoder = VQVAE_Decoder(points, embed_dim, nhead, num_layers)

        # Freeze encoder (quantization layer has no parameters)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()
        self.quantize.eval()

    def forward(self, images, P=None):
        img_feats = self.encoder(images)
        img_feats = img_feats.permute(0, 2, 3, 1)  # (N, H, W, C)

        quant, diff, encoding_inds, perplexity = self.quantize(img_feats)  # (N, H, W, C)
        N, H, W, C = quant.shape
        memory = quant.reshape(N, -1, C)

        ptclds = self.decoder(memory, P)
        return ptclds


    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.quantize.eval()

