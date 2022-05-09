import math
from re import M
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import utils


class MLP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels)
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
        QK_T = Q.matmul(K.permute(0, 2, 1))
       
        QK_T = QK_T.masked_fill(attention_mask.to(QK_T.device) == 0, -1e15)

        result = F.softmax(QK_T, dim=1).matmul(V)
        
        # print("Global Graph:", result.shape, result.min(), result.max(), result.mean())

        return result



class Sub_Graph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()

        self.hidden_size = hidden_size
        self.depth = depth

        self.g_enc = nn.ModuleList([
            nn.Sequential(
                MLP(hidden_size, hidden_size // 2)
            ) for i in range(depth)
        ])

    def forward(self, hidden_states, lengths):
        """
        hidden_states -> (num_polylines, max_num_vectors_per_polyline, hidden_size)
        lengths -> (num_polylines)
        """

        num_polylines, num_vectors, _ = hidden_states.shape
      
        lengths = torch.from_numpy(np.array(lengths)).to(hidden_states.device)        
        ones_index = torch.arange(num_vectors).repeat(num_polylines, 1).to(hidden_states.device) < lengths[:, None]

        for i in range(self.depth):

            hidden_states = self.g_enc[i](hidden_states)

            mask = torch.zeros_like(hidden_states, device=hidden_states.device)
            
            idx = ones_index.unsqueeze(2).repeat(1, 1, hidden_states.shape[2])
            mask[idx] = 1
            
            # testing the correctness of the mask
            # test_mask = torch.ones_like(hidden_states, device=hidden_states.device)
            
            # for j in range(len(lengths)):
            #     test_mask[j, lengths[j]:] = 0

            # assert (~(test_mask == mask)).sum() == 0, "Wrong mask" 

            agg = torch.max(hidden_states.masked_fill(mask == 0, -1e15), dim=1).values
            # agg = torch.max(hidden_states, dim=1).values

            agg = agg.unsqueeze(1)
            agg = agg.repeat(1, num_vectors, 1)
            hidden_states = torch.cat([hidden_states, agg], dim=2)
        
        hidden_states = torch.max(hidden_states, dim=1).values

        # print("SubGraph:", hidden_states.shape, hidden_states.min(), hidden_states.max(), hidden_states.mean())

        return hidden_states
