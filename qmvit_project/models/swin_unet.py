"""
Lightweight Swin-UNet style segmentation model implemented in PyTorch.
This is a simplified implementation intended for scaffolding and experimentation.
Replace with a production-grade implementation (e.g., from official repos) for best results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple multi-head self-attention used inside tiny Swin blocks
class SimpleSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, heads, head_dim)
        q = q.permute(0,2,1,3)  # (B, heads, N, head_dim)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, N, C)
        out = self.proj(out)
        return out

class TinySwinBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
    def forward(self, x):
        # x: (B, N, C)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def patchify(x, patch_size=4):
    # x: (B, C, H, W). Return (B, N, patch_dim)
    B,C,H,W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    ph = patch_size
    x = x.reshape(B, C, H//ph, ph, W//ph, ph)
    x = x.permute(0,2,4,3,5,1).reshape(B, (H//ph)*(W//ph), ph*ph*C)
    return x

def unpatchify(x, patch_size=4, H=128, W=128):
    # x: (B, N, patch_dim)
    B,N,PD = x.shape
    ph = patch_size
    C = PD // (ph*ph)
    h = H//ph; w = W//ph
    x = x.reshape(B,h,w,ph,ph,C).permute(0,5,1,3,2,4).reshape(B,C,H,W)
    return x

class TinySwinUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, img_size=128, patch_size=4, embed_dim=48):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # patch embedding
        self.patch_embed = nn.Linear(patch_size*patch_size*in_ch, embed_dim)
        # encoder blocks
        self.encoder1 = nn.Sequential(TinySwinBlock(embed_dim), TinySwinBlock(embed_dim))
        self.encoder2 = nn.Sequential(TinySwinBlock(embed_dim*2), TinySwinBlock(embed_dim*2))
        # simple down/up via linear projections
        self.down = nn.Linear(embed_dim, embed_dim*2)
        self.up = nn.Linear(embed_dim*2, embed_dim)
        # bottleneck
        self.bottleneck = nn.Sequential(TinySwinBlock(embed_dim*2))
        # decoder blocks
        self.decoder1 = nn.Sequential(TinySwinBlock(embed_dim), TinySwinBlock(embed_dim))
        # output head
        self.to_img = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_size*patch_size*out_ch)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B,C,H,W = x.shape
        x_p = patchify(x, self.patch_size)  # (B, N, P)
        x_e = self.patch_embed(x_p)  # (B, N, D)
        e1 = self.encoder1(x_e)
        # downsample: simple linear projection and reduce N by 2 via merging patches
        e2 = self.down(e1)  # dims changed
        b = self.bottleneck(e2)
        u = self.up(b)
        # skip connection add
        d = self.decoder1(u + e1)
        out_p = self.to_img(d)  # (B, N, patch_size*patch_size*out_ch)
        out = unpatchify(out_p, self.patch_size, H, W)
        return torch.sigmoid(out)
