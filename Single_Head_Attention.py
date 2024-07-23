import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# 单头自注意力的实现
class Attention(nn.Module):
    def __init__(self, dim_in, dim_q=2, dim_k=2, dim_v=3):
        super(Attention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_in = dim_in
        self.scale = dim_k ** -0.5
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, x):
        n, len, dim = x.shape  # n是batch大小, len是输入句子的长度, dim是每个token的维度
        # x --> (n, len, dim_in)
        q = self.q(x)  # q --> (n, len, dim_q)
        k = self.k(x)  # k --> (n, len, dim_k)
        v = self.v(x)  # v --> (n, len, dim_v)
        score = (q @ k.transpose(-1, -2)) * self.scale  # score --> (n, len, len)
        score = score.softmax(-1)

        out = score @ v  # out --> (n, len, dim_v)
        return out


attention = Attention(dim_in=6)
x = torch.randn(1, 512, 6)
y = attention(x)
print(x.shape)
print(y.shape)