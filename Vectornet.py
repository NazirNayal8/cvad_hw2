import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from GNNs import MLP, Global_Graph, Sub_Graph
from utils import (batch_init, get_dis_point_2_points, merge_tensors,
                   to_origin_coordinate)

global_input_hidden_scale = 1

class VectorNet(nn.Module):
    def __init__(self, hidden_size, device):
        super(VectorNet, self).__init__()

        self.sub_graph = Sub_Graph(hidden_size)
        self.global_graph = Global_Graph(hidden_size * global_input_hidden_scale)
        self.predict_traj = MLP(hidden_size * global_input_hidden_scale, 6*30*2 + 6)
        self.device = device
        self.hidden_size = hidden_size
        self.traj_completion_criterion = nn.SmoothL1Loss()

    def forward_encode_sub_graph(self, mapping, matrix, polyline_spans, batch_size):
        # len(input_list_list) = batch_size
        # len(input_list_list[i]) = #polylines in scene i
        # input_list_list[i][j] = #vectors in polyline j of scene i
        input_list_list = []
        for i in range(batch_size):
            input_list = []
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(
                    matrix[i][polyline_span], device=self.device)
                input_list.append(tensor)

            input_list_list.append(input_list)

        element_states_batch = []

        # Aim is getting a feature for each polyline
        #   from many vectors of it
        for batch_idx in range(batch_size):
            hidden_states, lengths = merge_tensors(
                input_list_list[batch_idx], self.device, self.hidden_size)
            # input_list_list[i] -> (8, 64), (6, 64), (19, 64)
            # hidden_states -> (3, 19, 64)

            hidden_states = self.sub_graph(hidden_states, lengths)
            # hidden_states.shape = (#polylines, hidden_size)
            element_states_batch.append(hidden_states)

        return element_states_batch

    def forward(self, mapping, validate=False):

        matrix = [i["matrix"] for i in mapping]
        polyline_spans = [i["polyline_spans"] for i in mapping]
        labels = [i["labels"] for i in mapping]

        batch_init(mapping)
        batch_size = len(matrix)

        element_states_batch = self.forward_encode_sub_graph(
            mapping, matrix, polyline_spans, batch_size)

        inputs, inputs_lengths = merge_tensors(
            element_states_batch, self.device, self.hidden_size * global_input_hidden_scale)
        # inputs.shape = (batch_size, max(#polylines), hidden_size)

        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros(
            [batch_size, max_poly_num, max_poly_num], device=self.device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask=attention_mask, mapping=mapping)

        outputs = self.predict_traj(hidden_states[:, 0, :])
        pred_probs = F.log_softmax(outputs[:, -6:], dim=-1)
        outputs = outputs[:, :- 6].view([batch_size, 6, 30, 2])

        ### YOUR CODE HERE ###
        loss = torch.tensor(0.0).float().to(outputs.device)
        for i in range(batch_size):
            gt_points = torch.from_numpy(np.array(labels[i]).reshape([30, 2])).float().to(outputs.device)
            
            # argmin = torch.argmax(pred_probs[i], dim=0).type(torch.LongTensor).to(outputs.device)  # find prediction with closest endpoint to gt
            
            #print(outputs[i, -1].shape, gt_points[-1].shape)
            
            argmin = torch.argmin(torch.sum((gt_points[-1].unsqueeze(0).repeat(outputs[i].shape[0], 1) - outputs[i,:,-1]) ** 2))

            loss += F.smooth_l1_loss(outputs[i, argmin].float(), gt_points).float()
            loss += F.nll_loss(pred_probs[i].float(), argmin).float()

        loss /= batch_size
            
        ### YOUR CODE HERE ###

        if validate:
            outputs = np.array(outputs.tolist())
            pred_probs = np.array(pred_probs.tolist(
            ), dtype=np.float32) if pred_probs is not None else pred_probs
            for i in range(batch_size):
                for each in outputs[i]:
                    to_origin_coordinate(each, i)

            return outputs, pred_probs

        return loss
