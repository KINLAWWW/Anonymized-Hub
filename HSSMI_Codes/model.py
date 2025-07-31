import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
import math

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: tuple):
    B, T, H, W, C = x.shape
    T_w, H_w, W_w = window_size 
    x = x.view(B, T // T_w, T_w, H // H_w, H_w, W // W_w, W_w, C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, T_w, H_w, W_w, C)
    
    return windows

def window_reverse(windows, window_size: tuple, T: int, H: int, W: int):
    T_w, H_w, W_w = window_size  
    num_windows = (T * H * W) // (H_w * W_w * T_w)
    B = windows.shape[0] // num_windows
    C = windows.shape[-1]
    x = windows.view(B, num_windows, T_w, H_w, W_w, C)
    x = x.view(B, T // T_w, H // H_w, W // W_w, T_w, H_w, W_w, C)
    x = x.permute(0, 1, 2, 4, 3, 5, 6, 7).contiguous()
    x = x.view(B, T, H, W, C)
    
    return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=(16, 3, 3), in_c=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()  
        x = self.proj(x)
        _, embed_dim, T_p, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x) 
        return x, T_p, H_p, W_p


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 4 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, T, H, W):
        B, L, C = x.shape
        assert L == T * H * W, "input feature has wrong size"

        x = x.view(B, T, H, W, C)
        pad_input = (T % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, T % 2))  
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size 
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape  
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) 
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = relative_position_bias.unsqueeze(0)
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]  
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(4,2,2), shift_size=(2,1,1),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert all(0 <= shift < window for shift, window in zip(self.shift_size, self.window_size)), "shift_size must be less than window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        T, H, W = self.T, self.H, self.W
        B, L, C = x.shape
        assert L == T * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        c = x.shape[-1]
        x = x.reshape(B, C, H, W, T) 
        w_t, w_h, w_w = self.window_size
        pad_t = (w_t - T % w_t) % w_t  
        pad_h = (w_h - H % w_h) % w_h
        pad_w = (w_w - W % w_w) % w_w

        x = F.pad(x, (0, pad_t, 0, pad_w, 0, pad_h))
        x = x.permute(0, 4, 2, 3, 1)
        _, Tp, Hp, Wp, _ = x.shape
    
        if any(shift > 0 for shift in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size) 
        x_windows = x_windows.view(-1, w_t * w_h * w_w, C)  
        attn_windows = self.attn(x_windows, mask=attn_mask)  
        attn_windows = attn_windows.view(-1, w_t, w_h, w_w, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Tp, Hp, Wp) 

        if any(shift > 0 for shift in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :T, :H, :W, :].contiguous()
        
        x = x.view(B, T * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.wt, self.wh, self.ww = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = (self.wt//2, self.wh//2, self.ww//2)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, T, H, W):
        Tp = int(np.ceil(T / self.wt)) * self.wt
        Hp = int(np.ceil(H / self.wh)) * self.wh
        Wp = int(np.ceil(W / self.ww)) * self.ww
        img_mask = torch.zeros((1,Tp, Hp, Wp, 1), device=x.device)
        t_slices = (slice(0, -self.wt),
                    slice(-self.wt, -self.wt),
                    slice(-self.wt, None))
        h_slices = (slice(0, -self.wh),
                    slice(-self.wh, -self.wh),
                    slice(-self.wh, None))
        w_slices = (slice(0, -self.ww),
                    slice(-self.ww, -self.ww),
                    slice(-self.ww, None))
        cnt = 0
        for t in t_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, t, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size) 
        mask_windows = mask_windows.view(-1, self.wt * self.wh * self.ww) 
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) 
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, T, H, W):
        if isinstance(x, int):
            x = torch.tensor(x).float().to(device) 
        attn_mask = self.create_mask(x, T, H, W)  
        for blk in self.blocks:
            blk.T, blk.H, blk.W = T, H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x,T, H, W)
            T, H, W =(T + 1) // 2, (H + 1) // 2, (W + 1) // 2

        return x, T, H, W


class SwinMI(nn.Module):
    def __init__(self, patch_size=(16,3,3), in_chans=1, num_classes=2,
                 embed_dim=96, depths=(2, 6, 4), num_heads=(3, 6, 8),
                 window_size=(3,3,3), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 4 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=int(embed_dim * 4 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        x, T, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, T, H, W = layer(x, T, H, W)
        x = self.norm(x)  
        x = self.avgpool(x.transpose(1, 2)) 
        y = torch.flatten(x, 1) 
        x = self.head(y)

        return x
