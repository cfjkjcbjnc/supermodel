import torch
import math
from torch import nn, einsum
import torch.nn.functional as F
from model.rmsnorm import RMSNorm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class RectifiedLinearAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., rmsnorm=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.norm = RMSNorm(inner_dim) if rmsnorm else nn.LayerNorm(inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = F.relu(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out =  self.to_out(self.norm(out))
        return out

import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChannelAttention, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(nn.Linear(out_channel, out_channel // 4, bias=False), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(out_channel // 4, out_channel, bias=False), nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(out_channel))

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.conv(x)
        y1 = self.pool(x)
        y1 = y1.reshape((n, -1))
        y1 = self.linear1(y1)
        y1 = self.linear2(y1)
        y1 = y1.reshape((n, self.out_channel, 1, 1))

        y1 = y1.expand_as(x).clone()
        y = x * y1
        return F.relu(y + self.conv2(y))


if __name__ == "__main__":
    rla = RectifiedLinearAttention(192, 3, 64, 0., True)
    x = torch.ones([2, 196, 192])
    attn = rla(x)
    print(attn.shape)

