import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = x + self.pos_embedding
        out = self.dropout(out)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = nn.Linear(in_dim, self.head_dim * self.heads)
        self.key = nn.Linear(in_dim, self.head_dim * self.heads)
        self.value = nn.Linear(in_dim, self.head_dim * self.heads)
        self.out = nn.Linear(self.head_dim * self.heads, in_dim)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        b = x.shape[0]
        q = self.query(x).view(b, -1, self.heads, self.head_dim)
        k = self.key(x).view(b, -1, self.heads, self.head_dim)
        v = self.value(x).view(b, -1, self.heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)
        v = v.transpose(1, 2)
        
        score = torch.matmul(q, k) / self.scale
        attn = torch.softmax(score, dim=3)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, -1, self.heads * self.head_dim)
        
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(in_dim)
        self.ff = FeedForwardBlock(in_dim, hid_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        out = self.dropout(out)
        out += residual
        residual = out
        out = self.norm2(out)
        out = self.ff(out)
        out += residual
        return out
    
class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, hid_dim, num_layers=8, num_heads=8, dropout_rate=0.1, attn_dropout_rate=0.0):
        super().__init__()
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EncoderBlock(in_dim, hid_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        out = self.pos_embedding(x)
        for layer in self.encoder_layers:
            out = layer(out)
        out = self.norm(out)
        return out
    
class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=(32, 32),
                 patch_size=(4, 4),
                 emb_dim=512,
                 hid_dim=512 * 2,
                 num_heads=8,
                 num_layers=6,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 ):
        super().__init__()
        h, w = image_size

        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

    def forward(self, x):
        emb = self.embedding(x)     
        emb = emb.permute(0, 2, 3, 1) 
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        feat = self.transformer(emb)
        return feat[:, 0]



    