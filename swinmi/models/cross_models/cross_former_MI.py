import torch
import torch.nn as nn
from einops import rearrange, repeat
from math import ceil

from .cross_encoder import Encoder
from .cross_embed import DSW_embedding

class CrossformerForMI(nn.Module):
    """
    Crossformer -> adapted for MI EEG classification.

    Expected input:
      - Either (B, T, C) where T = timepoints, C = channels
      - Or (B, C, T) if you prefer channels-first (set channels_first=True)

    Output:
      - logits (B, num_classes)
    """
    def __init__(
        self,
        data_dim,       # number of EEG channels (C)
        in_len,         # number of timepoints (T)
        seg_len,        # segment length for embedding
        num_classes=2,  # number of MI classes
        channels_first=False,  # if True, expect input (B, C, T). Else (B, T, C)
        win_size=4,
        factor=10,
        d_model=512,
        d_ff=1024,
        n_heads=8,
        e_layers=3,
        dropout=0.0,
        baseline=False,
        device=torch.device('cuda:0'),
        pooling='avg'   # 'avg' | 'max' | 'attn'
    ):
        super(CrossformerForMI, self).__init__()

        self.data_dim = data_dim
        self.in_len = in_len
        self.seg_len = seg_len
        self.channels_first = channels_first
        self.pooling = pooling
        self.device = device
        self.baseline = baseline

        # pad to integer number of segments (same as original)
        self.pad_in_len = int(ceil(in_len / seg_len) * seg_len)
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding: same segment-wise embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        # positional embedding shape: (1, data_dim, n_seg, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder (reuse original Encoder)
        self.encoder = Encoder(
            e_layers,
            win_size,
            d_model,
            n_heads,
            d_ff,
            block_depth=1,
            dropout=dropout,
            in_seg_num=self.pad_in_len // seg_len,
            factor=factor,
        )

        # Pooling head: if 'attn', we use a simple attention pooling
        if pooling == 'attn':
            self.attn_pool_q = nn.Parameter(torch.randn(1, 1, d_model))  # learnable query
            self.attn_softmax = nn.Softmax(dim=-1)

        # Classification head
        hidden = max(d_model // 2, 64)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        """
        x: if channels_first=False -> (B, T, C)
           if channels_first=True  -> (B, C, T)
        returns logits: (B, num_classes)
        """
        # ---- normalize input layout to (B, T, C) for embedding ----
        if self.channels_first:
            # user passed (B, C, T) -> convert to (B, T, C)
            x = x.permute(0, 2, 1).contiguous()

        B, T, C = x.shape
        assert C == self.data_dim, f"expected data_dim={self.data_dim}, but got C={C}"
        assert T == self.in_len, f"expected in_len={self.in_len}, but got T={T}"

        # optional baseline: compute and subtract (trend removal)
        if self.baseline:
            base = x.mean(dim=1, keepdim=True)  # (B,1,C)
            x = x - base
        else:
            base = None

        # pad at time axis if needed (prepend first value like original)
        if self.in_len_add > 0:
            pad_part = x[:, :1, :].expand(-1, self.in_len_add, -1)  # (B, in_len_add, C)
            x = torch.cat((pad_part, x), dim=1)  # (B, pad_in_len, C)

        # embedding: DSW_embedding expects (B, T, C) and returns (B, C, n_seg, d_model)
        x_emb = self.enc_value_embedding(x)  # (B, C, n_seg, d_model)
        # add pos embedding
        x_emb = x_emb + self.enc_pos_embedding  # broadcast
        # layernorm on last dim
        x_emb = self.pre_norm(x_emb)

        # encoder forward: expect (B, C, n_seg, d_model) and returns same shape
        enc_out = self.encoder(x_emb)  # (B, C, n_seg, d_model)
        if isinstance(enc_out, list):
            enc_out = enc_out[-1]
        # pooling to produce single vector per sample
        # combine channel and segment dims (C * n_seg) -> pool over them
        B, C, n_seg, d_model = enc_out.shape
        # reshape to (B, C*n_seg, d_model)
        flat = enc_out.view(B, C * n_seg, d_model)

        if self.pooling == 'avg':
            rep = flat.mean(dim=1)  # (B, d_model)
        elif self.pooling == 'max':
            rep, _ = flat.max(dim=1)
        elif self.pooling == 'attn':
            # simple attention pooling with learnable query
            # q: (1,1,d_model) -> (B,1,d_model)
            q = self.attn_pool_q.expand(B, -1, -1)  # (B,1,d_model)
            # compute attention scores: q · k^T  -> (B,1, L)
            scores = torch.matmul(q, flat.transpose(1, 2)) / (d_model ** 0.5)  # (B,1,L)
            weights = self.attn_softmax(scores)  # (B,1,L)
            rep = torch.matmul(weights, flat).squeeze(1)  # (B, d_model)
        else:
            raise ValueError("pooling must be 'avg'|'max'|'attn'")

        # optional add baseline (if you want to include trend info)
        if self.baseline and base is not None:
            # base shape: (B,1,C) -> embed to d_model by simple mean+linear (optional)
            # Here we simply ignore base, or you could project base to d_model and add.
            pass

        logits = self.classifier(rep)  # (B, num_classes)
        return logits
