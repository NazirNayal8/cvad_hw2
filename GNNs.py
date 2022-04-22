import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import utils


class Global_Graph(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        pass

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        pass


class Sub_Graph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()

    def forward(self, hidden_states, lengths):
        pass
