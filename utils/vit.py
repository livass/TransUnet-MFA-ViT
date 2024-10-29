import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x
        
class MOA(nn.Module):
    def __init__(self,hidden_dim, H, W):
        super(MOA, self).__init__()
        self.kernel_size = 3
        self.k2=5
        self.padding=1
        self.padding2=2
        self.H = H
        self.W = W
        self.v_pj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 3**4)
        self.unfold = nn.Unfold(kernel_size=3,padding=1,stride=1)
        self.fold = nn.Fold(output_size=(H, W), kernel_size=3,padding=1)
        self.attn2=nn.Linear(hidden_dim, 5**4)
        self.unfold2 = nn.Unfold(kernel_size=5,padding=2,stride=1)
        self.fold2 = nn.Fold(output_size=(H, W), kernel_size=5,padding=2)
        self.fu=nn.Linear(hidden_dim*2, hidden_dim)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=5, stride=1,padding=2)
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(p=0.5)
        self.bn1=nn.BatchNorm2d(hidden_dim)
        self.bn2=nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v_pj(x)
        v=v.permute(0, 3, 2, 1)
        v = self.unfold(v)
        x3=x
        x_max=x3
        v=v.reshape(B,C, self.kernel_size**2, H*W)
        v=v.permute(0, 3, 2, 1)
        a = self.attn(x).reshape(B,H*W, self.kernel_size**2, self.kernel_size**2)
        a = torch.softmax(a, dim=-1)
        
        v2=self.v_pj(x)
        v2=v2.permute(0, 3, 2, 1)
        v2=self.unfold2(v2)

        v2=v2.reshape(B,C, self.k2**2, H*W)
        v2=v2.permute(0, 3, 2, 1)
        
        a2= self.attn2(x).reshape(B,H*W, self.k2**2, self.k2**2)
        a2 = torch.softmax(a2, dim=-1)
        
        x = torch.matmul(a, v)
        #x=self.bn1(x)
        x=x.permute(0, 3,2,1)
        x=x.reshape(B, C*self.kernel_size*self.kernel_size,H*W)
        x = self.fold(x).permute(0,3,2,1)
        x4=x
        x=self.relu(x)
        
        
        x_max=x_max.permute(0, 3, 2, 1)
        x_max1=self.max_pool1(x_max)
        x_max1=x_max1.permute(0, 3, 2, 1)
        
        x=x+x_max1
        x=self.relu(x)
        x2 = torch.matmul(a2, v2)
        #x2=self.bn2(x2)
        x2=x2.permute(0, 3,2,1)
        x2=x2.reshape(B, C*self.k2*self.k2,H*W)
        x2 = self.fold2(x2).permute(0,3,2,1)
        x2=self.relu(x)
        x_max2=self.max_pool2(x_max)
        x_max2=x_max2.permute(0, 3, 2, 1)
        x2=x2+x_max2
        x2=self.relu(x2)
        x_fu=torch.cat([x, x2], dim=3)
        
        x_fu=self.fu(x_fu)
        x_fu=self.relu(x_fu)
        x=x3+x_fu
        x=x.permute(0, 3,2,1)
        x=self.bn1(x)
        x=x.permute(0, 2,3,1)
        return x

class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.MOA=MOA(in_channels * (patch_dim ** 2),img_dim // patch_dim,img_dim // patch_dim)
        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape
  
        #-----------------------
        #H = int(tokens**0.5)
        #W = int(tokens**0.5)
        #img_patches = img_patches.reshape(batch_size, H, W, _)
        #img_patches=self.MOA(img_patches)
        #img_patches = img_patches.reshape(batch_size, tokens, _)
        project = self.projection(img_patches)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)
 
        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[:tokens + 1, :]
        
        x = self.dropout(patches)

        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x


if __name__ == '__main__':
    vit = ViT(img_dim=128,
              in_channels=3,
              patch_dim=16,
              embedding_dim=512,
              block_num=6,
              head_num=4,
              mlp_dim=1024)
    print(sum(p.numel() for p in vit.parameters()))
    print(vit(torch.rand(1, 3, 128, 128)).shape)
