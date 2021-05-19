import torch
from torch import nn
from torch.nn import functional as F
from models.backbone import build_backbone
from models.pointalign import SmallDecoder

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

        embed = torch.randn(embed_dim, num_embed)
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

        # do we need to return encoding_inds, encodings??
        return quantized, diff, encoding_inds, perplexity
    

    def quantize(self, embed_ind):
        return F.embedding(embed_ind, self.embed.T)


class VQVAE(nn.Module):

    def __init__(self, points=None, num_embed=256):
        super().__init__()

        self.encoder, feat_dims = build_backbone('resnet18', pretrained=True)
        self.quantize = VectorQuantizer(num_embed, embed_dim=512)
        self.decoder = SmallDecoder(points, img_feat_dim=512, hidden_dim=128)

    def forward(self, images, P=None):
        z = self.encoder(images)[-1]
        z = z.permute(0, 2, 3, 1)
        
        quant, diff, encoding_inds, perplexity = self.quantize(z)
        quant = quant.permute(0, 3, 1, 2)  # (N, C, H, W)
        l_vq = diff.unsqueeze(0)

        ptclds = self.decoder(quant, P)

        return ptclds, l_vq, encoding_inds, perplexity


