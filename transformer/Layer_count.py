import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
import Utils
from transformer.Layers import EncoderLayer
import sys

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).any(dim=-1).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q): # MODIFY LATER
    """ For masking out the padding part of key sequence. """
    seq_k = seq_k.unsqueeze(0)  # Shape: (1, L, K)
    seq_q = seq_q.unsqueeze(0)  # Shape: (1, L, K)

    # Expand to fit the shape of the key-query attention matrix
    len_q = seq_q.size(1)  # Sequence length for the query
    padding_mask = seq_k.sum(dim=2).eq(0)  # Identify rows with all zeros in seq_k (shape: 1, L)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # Shape: (1, lq, lk)

    return padding_mask


def get_subsequent_mask(count_matrix):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    count_matrix = count_matrix.unsqueeze(0)
    sz_b, len_s, _ = count_matrix.size() # [1, L]

    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=count_matrix.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask
    
class Encoder_count(nn.Module):
    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, bin_size, num_batches = 1):
        super().__init__()

        self.num_batches = num_batches
        self.d_model = d_model

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])
        
        self.event_conv_filter = nn.Conv1d(
            in_channels=num_types,  
            out_channels=d_model,  
            kernel_size=1
        )

    def positional_enc(self, count_L):
        idx = torch.arange(1, count_L + 1, device=torch.device('cuda')).unsqueeze(0).repeat(self.num_batches, 1) # Shape (num_batches, L)
        position_vec = torch.tensor(
            [2 * math.pow(count_L, 2.0 * (i // 2) / self.d_model) for i in range(self.d_model)],
            device=torch.device('cuda'))
        result = idx.unsqueeze(-1) / position_vec
        
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        
        return result
    
    def count_enc(self, count_matrix):
        count_matrix = count_matrix.unsqueeze(0).permute(0, 2, 1)  # Shape: (1, K, L)
    
        conv_enc = self.event_conv_filter(count_matrix).permute(0, 2, 1)  # Shape: (1, L, d_model)
        return conv_enc
        
    
    def forward(self, count_matrix, non_pad_mask):
        """ Encode count sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        
        count_L = count_matrix.size(0)
        
        slf_attn_mask_subseq = get_subsequent_mask(count_matrix)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=count_matrix, seq_q=count_matrix)
        
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        
        pos_enc = self.positional_enc(count_L)
        enc_output = self.count_enc(count_matrix=count_matrix)
        
        for enc_layer in self.layer_stack:
            enc_output += pos_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output
