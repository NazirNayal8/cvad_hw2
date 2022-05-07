import math
from re import M

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import utils


class MLP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        return self.layer(x)

class Concat(nn.Module):

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.concat(x, dim=self.dim)


class MaxPool(nn.Module):

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.max(x, dim=self.dim).values

class Global_Graph(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, hidden_states, attention_mask=None, mapping=None):
        """
        hidden_states -> (batch_size, max_num_polylines, hidden_size)
        attention_mask -> (batch_size, max_num_polylines, max_num_polylines)
        """
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.key(hidden_states)

        # fill masked positions with very negative numbers
        QK_T = Q.matmul(K.T).masked_fill(attention_mask.type(torch.BoolTensor), -1e8)

        return F.softmax(QK_T, dim=1).matmul(V)



class Sub_Graph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()

        self.hidden_size = hidden_size
        self.depth = depth
        
        # a single layer MLP shared for all nodes, followed by layer normalization and ReLU
        self.g_enc = nn.ModuleList([
            nn.Sequential(
                MLP((2 ** i) * hidden_size, (2 ** i) * hidden_size)
            ) for i in range(depth)
        ])


    def forward(self, hidden_states, lengths):
        """
        hidden_states -> (num_polylines, max_num_vectors_per_polyline, hidden_size)
        lengths -> (num_polylines)
        """

        num_polylines, num_vectors, _ = hidden_states.shape

        

        for i in range(self.depth):

            hidden_states = self.g_enc[i](hidden_states)

            mask = torch.ones_like(hidden_states, device=hidden_states.device)
            for i in range(len(lengths)):
                mask[i, lengths[i]:] = -1e8

            agg = torch.max(hidden_states * mask, dim=1).values
            agg = agg.unsqueeze(1)
            agg = agg.repeat(1, num_vectors, 1)
            hidden_states = torch.cat([hidden_states, agg], dim=2)
        
        return torch.max(hidden_states, dim=1).values
