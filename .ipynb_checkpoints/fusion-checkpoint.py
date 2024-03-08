import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_


class CrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim):
        super(CrossAttention, self).__init__()
        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)

    def forward(self, input_a, input_b):
        # 线性映射
        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)
        y = mapped_b.transpose(1, 2)
        # 计算注意力权重
        scores = torch.matmul(mapped_a, mapped_b.transpose(1, 2))  # (batch_size, seq_len_a, seq_len_b)
        attentions_a = torch.softmax(scores, dim=-1)  # 在维度2上进行softmax，归一化为注意力权重 (batch_size, seq_len_a, seq_len_b)
        attentions_b = torch.softmax(scores.transpose(1, 2),
                                     dim=-1)  # 在维度1上进行softmax，归一化为注意力权重 (batch_size, seq_len_b, seq_len_a)
        # 使用注意力权重来调整输入表示
        output_a = torch.matmul(attentions_b, input_b)  # (batch_size, seq_len_a, input_dim_b)
        output_b = torch.matmul(attentions_a.transpose(1, 2), input_a)  # (batch_size, seq_len_b, input_dim_a)
        return output_a + input_a, output_b + input_b


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads=16, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        head_dim = d_model // heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x_rgb, x_tir):
        B, N, C = x_rgb.shape
        q = self.q(x_tir).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = self.k(x_rgb).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = self.v(x_rgb).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Fusion_Module(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fusion_layer = nn.Linear(2 * d_model, d_model)
        self.layer1 = nn.LayerNorm(d_model)
        self.sig = nn.GELU()
        self.fusin_attn = MultiHeadAttention(d_model)
        self.fusin_attn1 = MultiHeadAttention(d_model)
        self.fusin_attn2 = MultiHeadAttention(d_model)
        self.fusin_attn3 = MultiHeadAttention(d_model)
        self.drop_path = nn.Dropout(0.02)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

        self.mlp = FeedForward(d_model, d_model * 4, 0.1)
        self.mlp1 = FeedForward(d_model, d_model * 4, 0.1)

    def forward(self, x_rgb, x_tir, x_sum):
        # rgb_cross =  self.drop_path(self.cross(self.norm1(x_rgb), self.norm2(x_tir)))
        x_sum_att = self.drop_path(self.fusin_attn3(self.norm1(x_sum), self.norm1(x_sum)))
        x_sum = x_sum + x_sum_att
        x_sum = x_sum + self.drop_path(self.mlp1(self.norm3(x_sum)))

        rgb_fusion = self.drop_path(self.fusin_attn(self.norm4(x_rgb), self.norm5(x_tir))) + self.drop_path(
            self.fusin_attn(self.norm4(x_rgb), x_sum))
        tir_fusion = self.drop_path(self.fusin_attn2(self.norm5(x_tir), self.norm4(x_rgb))) + self.drop_path(
            self.fusin_attn(self.norm5(x_tir), x_sum))
        res = rgb_fusion + tir_fusion + x_sum
        res = res + self.drop_path(self.mlp(self.norm2(res)))

        # res = self.drop_path(self.fusin_attn3(res, res)) + res
        # res = res + self.drop_path(self.mlp1(self.norm5(res)))
        return res, rgb_fusion, tir_fusion


class Block_fusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fusion = Fusion_Module(d_model)

    def forward(self, x_rgb, x_tir):
        x_sum = x_rgb + x_tir
        for i in range(1):
            x_sum, x_rgb_one, x_tir_one = self.fusion(x_rgb, x_tir, x_sum)

        return x_sum


class Fusion_Module_all(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fusion_layer = nn.Linear(d_model, 2 * d_model)
        self.fusion_layer1 = nn.Linear(2 * d_model, 4 * d_model)
        self.fusion_layer2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x_before):
        return self.fusion_layer2(self.fusion_layer1(self.fusion_layer(x_before))) + x_before