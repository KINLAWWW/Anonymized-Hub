import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np


def drop_path_f(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) layer."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Partition input tensor into 3D windows.

    Args:
        x: (B, T, H, W, C)
        window_size: (T_w, H_w, W_w)

    Returns:
        windows: (num_windows*B, T_w, H_w, W_w, C)
    """
    B, T, H, W, C = x.shape
    T_w, H_w, W_w = window_size
    x = x.view(B, T // T_w, T_w, H // H_w, H_w, W // W_w, W_w, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, T_w, H_w, W_w, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int, int], T: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse windows into original feature map.

    Args:
        windows: (num_windows*B, T_w, H_w, W_w, C)
        window_size: (T_w, H_w, W_w)
        T, H, W: original dimensions

    Returns:
        x: (B, T, H, W, C)
    """
    T_w, H_w, W_w = window_size
    num_windows = (T * H * W) // (T_w * H_w * W_w)
    B = windows.shape[0] // num_windows
    C = windows.shape[-1]
    x = windows.view(B, num_windows, T_w, H_w, W_w, C)
    x = x.view(B, T // T_w, H // H_w, W // W_w, T_w, H_w, W_w, C)
    x = x.permute(0, 1, 2, 4, 3, 5, 6, 7).contiguous()
    x = x.view(B, T, H, W, C)
    return x


class PatchEmbed(nn.Module):
    """3D patch embedding."""
    def __init__(self, patch_size=(16,3,3), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        # Input: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, H, W)
        x = self.proj(x)
        _, C, T_p, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, T_p, H_p, W_p


class PatchMerging(nn.Module):
    """Patch merging layer for 3D data."""
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(8*dim, 4*dim, bias=False)
        self.norm = norm_layer(8*dim)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == T * H * W
        x = x.view(B, T, H, W, C)
        pad_input = (T%2==1) or (H%2==1) or (W%2==1)
        if pad_input:
            x = F.pad(x, (0,0,0,W%2,0,H%2,0,T%2))
        x0 = x[:,0::2,0::2,0::2,:]
        x1 = x[:,1::2,0::2,0::2,:]
        x2 = x[:,0::2,1::2,0::2,:]
        x3 = x[:,1::2,1::2,0::2,:]
        x4 = x[:,0::2,0::2,1::2,:]
        x5 = x[:,1::2,0::2,1::2,:]
        x6 = x[:,0::2,1::2,1::2,:]
        x7 = x[:,1::2,1::2,1::2,:]
        x = torch.cat([x0,x1,x2,x3,x4,x5,x6,x7], dim=-1)
        x = x.view(B,-1,8*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Mlp(nn.Module):
    """MLP layer used in transformer blocks."""
    def __init__(self, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, act_layer=nn.GELU, drop: float=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    """3D window-based multi-head self-attention (W-MSA) module."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Th, H, W)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        num_relative_positions = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, num_heads))

        # Compute relative position index
        coords_t = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w))  # [3, Th, H, W]
        coords_flatten = coords.flatten(1)  # [3, Th*H*W]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 3]

        # Shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        self.register_buffer("relative_position_index", relative_coords.sum(-1))  # [N, N]

        # QKV projection layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input features with shape (num_windows*B, N, C)
            mask: Optional attention mask (num_windows, N, N)
        """
        B_, N, C = x.shape
        # QKV projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale

        # Compute attention
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        ).permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, nH, N, N]
        attn = attn + relative_position_bias

        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Attention output
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """3D Swin Transformer Block with W-MSA or SW-MSA."""

    def __init__(self, dim, num_heads, window_size=(4, 2, 2), shift_size=(2, 1, 1),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert all(0 <= s < w for s, w in zip(shift_size, window_size)), "shift_size must be less than window_size"

        # Layers
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        B, L, C = x.shape
        T, H, W = self.T, self.H, self.W
        assert L == T * H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, C, H, W, T)  # Reorder channels for padding

        # Padding to multiples of window size
        pad_t = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, pad_t, 0, pad_w, 0, pad_h))
        x = x.permute(0, 4, 2, 3, 1)  # [B, T, H, W, C]

        Tp, Hp, Wp = x.shape[1:4]

        # Cyclic shift for SW-MSA
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size).view(-1, 
                               self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        # Attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Tp, Hp, Wp)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x

        # Remove padding
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :T, :H, :W, :].contiguous()

        x = x.view(B, T * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """A Swin Transformer stage consisting of multiple SwinTransformerBlocks."""

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = tuple(w // 2 for w in window_size)
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, num_heads, window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.downsample = downsample(dim, norm_layer=norm_layer) if downsample else None

    def create_mask(self, x, T, H, W):
        """Create attention mask for SW-MSA."""
        Tp = int(np.ceil(T / self.window_size[0])) * self.window_size[0]
        Hp = int(np.ceil(H / self.window_size[1])) * self.window_size[1]
        Wp = int(np.ceil(W / self.window_size[2])) * self.window_size[2]

        img_mask = torch.zeros((1, Tp, Hp, Wp, 1), device=x.device)
        cnt = 0
        for t_slice in [(0, -self.window_size[0]), (-self.window_size[0], None)]:
            for h_slice in [(0, -self.window_size[1]), (-self.window_size[1], None)]:
                for w_slice in [(0, -self.window_size[2]), (-self.window_size[2], None)]:
                    img_mask[:, slice(*t_slice), slice(*h_slice), slice(*w_slice), :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size).view(-1, np.prod(self.window_size))
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x, T, H, W):
        attn_mask = self.create_mask(x, T, H, W)
        for blk in self.blocks:
            blk.T, blk.H, blk.W = T, H, W
            if self.use_checkpoint and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample:
            x = self.downsample(x, T, H, W)
            T, H, W = (T + 1) // 2, (H + 1) // 2, (W + 1) // 2

        return x, T, H, W
        

class SwinMI(nn.Module):
    """Swin Transformer for 3D EEG (SwinMI)."""
    def __init__(self,
                 patch_size=(16,3,3),
                 in_chans=1,
                 num_classes=2,
                 embed_dim=96,
                 depths=(2,2,4,2),
                 num_heads=(2,2,4,6),
                 window_size=(4,2,2),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 visual=False,
                 use_checkpoint=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.visual = visual
        self.num_features = int(embed_dim * 4 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim*4**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers-1 else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of SwinMI.

        Args:
            x: (B, C, T, H, W)
        Returns:
            logits: (B, num_classes)
            optionally features: (B, C)
        """
        x = x.permute(0,2,3,4,1)  # to (B, T, H, W, C)
        x, T, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, T, H, W = layer(x, T, H, W)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1,2))
        y = torch.flatten(x,1)
        x = self.head(y)
        if self.visual:
            return x, y
        return x