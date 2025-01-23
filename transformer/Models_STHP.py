import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
import Utils
import transformer.Layer_event as Layer_event
import transformer.Layer_count as Layer_count
from transformer.Layer_count import Encoder_count
from transformer.Layer_event import Encoder_event
import sys

class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out

class SparseTransformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, lambda_window=1000, bin_size=5):
        super().__init__()

        self.num_types = num_types
        self.bin_size = bin_size
        self.d_model = d_model
        self.dropout = dropout
        
        self.count_encoder = Encoder_count(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            bin_size=bin_size
        )
        self.event_encoder= Encoder_event(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            lambda_window = lambda_window
        )

        self.linear_event = nn.Linear(d_model, num_types)
        self.linear_count = nn.Linear(d_model, num_types)
        
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)
        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)
        
        # self.linear_match_event = None
    
    def get_theta_matrix(self, time, count_enc_output):
        idx_row = (torch.ceil(time / self.bin_size) - 1.0).to(device=time.device, dtype=torch.long).squeeze()
        
        theta_matrix = count_enc_output[:, idx_row, :]
        return theta_matrix
    
    def get_combined_enc_output(self, event_enc_output, count_enc_output, event_time):
        count_enc_output = self.get_theta_matrix(time=event_time, count_enc_output=count_enc_output)
        return count_enc_output + event_enc_output
        
        # count_enc_output = count_enc_output.transpose(1, 2)
        # count_L = count_enc_output.size(2)
        # num_samples = event_enc_output.size(1)
        
        # print("EVENT")
        # print(event_enc_output.shape)
        # print(count_enc_output.shape)
        # print(num_samples, count_L)
        # sys.exit()
        
        # device = count_enc_output.device
        
        # if self.linear_match_event is None:
        #     self.linear_match_event = nn.Linear(count_L, num_samples).to(device)
            
        # count_enc_output = self.linear_match_event(count_enc_output).transpose(1, 2)
        # return count_enc_output + event_enc_output
    
    def forward(self, count_matrix, event_type, event_time, lambda_window = 1000):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        
        non_pad_mask_event = Layer_event.get_non_pad_mask(event_type)
        non_pad_mask_count = Layer_count.get_non_pad_mask(count_matrix)
        
        event_enc_output = self.event_encoder(event_type, event_time, non_pad_mask_event)
        count_enc_output = self.count_encoder(count_matrix, non_pad_mask_count)
        
        device = event_enc_output.device
        event_enc_output = event_enc_output.to(device)
        count_enc_output = count_enc_output.to(device)

        combined_enc_output = self.get_combined_enc_output(event_enc_output=event_enc_output, count_enc_output=count_enc_output, event_time=event_time)

        time_prediction = self.time_predictor(combined_enc_output, non_pad_mask_event)  
        type_prediction = self.type_predictor(combined_enc_output, non_pad_mask_event)
    
        return combined_enc_output, (type_prediction, time_prediction)