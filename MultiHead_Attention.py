import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MultiHead_Attention(nn.Module):
    def __init__(self, dim_in, dmodel, num_head):
        super(MultiHead_Attention, self).__init__()
        self.dim_in = dim_in
        self.dmodel = dmodel
        self.num_head = num_head
        self.dhead = self.dmodel // self.num_head # must // , because integer type
        self.scale = self.dhead ** -0.5

        self.q = nn.Linear(dim_in, dmodel)
        self.k = nn.Linear(dim_in, dmodel)
        self.v = nn.Linear(dim_in, dmodel)
        self.final_proj = nn.Linear(dmodel, dmodel)

    def forward(self, x):
        batch, n, din = x.shape  # x --> (batch, n, din)
        q = self.q(x) # q --> (batch, n, dmodel)
        # multihead
        q = q.reshape(batch, n, self.num_head, self.dhead).transpose(1,2) # q --> (batch, num_head, n, dhead)
        k = self.k(x).reshape(batch, n, self.num_head, self.dhead).transpose(1,2) # k --> (batch, num_head, n, dhead)
        v = self.v(x).reshape(batch, n, self.num_head, self.dhead).transpose(1,2) # v --> (batch, num_head, n, dhead)

        attention_score = ((q @ k.transpose(-1,-2)) * self.scale).softmax(dim=-1) # (batch, num_head, n, n)

        out = (attention_score @ v).transpose(1,2).reshape(batch, n, self.dmodel) # (batch, n, dmodel)

        out  = self.final_proj(out)

        return out # (batch, n, dmodel)


mhsa = MultiHead_Attention(dim_in=10, dmodel=6, num_head=3)
x = torch.randn((1, 4, 10))
y = mhsa(x)
print(x.shape)
print(y.shape)