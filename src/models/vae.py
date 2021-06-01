import torch
from torch import nn
from torch.nn import functional as F
from models.backbone import build_backbone
from models.pointalign import SmallDecoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class VAE(nn.Module):

    def __init__(self, points=None, latent_dim=8, hidden_dim=128):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder, feat_dims = build_backbone('resnet18', pretrained=True)
        self.fc_mu = nn.Linear(feat_dims[-1] * 25, latent_dim)
        self.fc_logvar = nn.Linear(feat_dims[-1] * 25, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, feat_dims[-1] * 25)
        self.decoder = SmallDecoder(points, input_dim=feat_dims[-1], hidden_dim=hidden_dim)


    def latent_sample(self, mu, logvar):
        eps = torch.randn_like(logvar)
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + eps * std
        else:
            return eps


    def decode(self, latents, P=None):
        bs = latents.shape[0]
        decoder_input = self.fc_decode(latents)
        decoder_input = decoder_input.view((bs, -1, 5, 5))
        return self.decoder(decoder_input, P)


    def forward(self, images, P=None):
        bs = images.shape[0]
        img_feats = self.encoder(images)[-1]  # (N, C, H, W)

        flat_feats = img_feats.view(bs, -1)
        mu = self.fc_mu(flat_feats)
        logvar = self.fc_logvar(flat_feats)
        sample = self.latent_sample(mu, logvar)

        ptclds = self.decode(sample, P)

        return ptclds, mu, logvar

