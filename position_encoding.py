import torch
import torch.nn as nn
import math



class Postional_Encoding(nn.Module):
    def __init__(self, d_model, max_seqlen=80) -> None:
        super().__init__()
        pe = torch.zeros(max_seqlen, d_model)
        for pos in range(max_seqlen):
            for i in range(start=0, stop=d_model, step=2):
                pe[pos, i] = math.sin(pos / (10000**(i/(2*d_model))))
                pe[pos, i+1] = math.cos(pos / (10000**(i/(2*d_model))))
        pe = pe.unsqueeze(0) # (seq_len, d_model) -> (1, seq_len, d_model), 增加了batch_size这个维度方便后续广播机制操作 
        self.register_buffer('pe', pe)

    def forward(self, x): # x.shape --> (b, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x






