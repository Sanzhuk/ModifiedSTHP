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
    # Handle batched input for volume data
    return seq.sum(dim=-1).ne(0).float().unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    # Handle batched input
    batch_size = seq_k.size(0)
    len_q = seq_q.size(1)
    
    # Identify rows with all zeros in seq_k
    padding_mask = seq_k.sum(dim=-1).eq(0)  # Shape: (batch_size, L)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # Shape: (batch_size, lq, lk)
    
    return padding_mask


def get_subsequent_mask(volume_matrix):
    """ For masking out the subsequent info. """
    batch_size, len_s, _ = volume_matrix.size()
    
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=volume_matrix.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # batch x ls x ls
    
    return subsequent_mask
    
class Encoder_volume(nn.Module):
    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, bin_size_volume=3, num_batches = 1):
        super().__init__()

        self.num_batches = num_batches
        self.d_model = d_model

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])
        
        # Change the event embedding to a linear layer for volume data
        self.volume_proj = nn.Linear(1, d_model)  # Project volume data to d_model dimensions

    def positional_enc(self, count_L):
        """
        Generate positional encoding for a given length count_L
        Args:
            count_L: Length of the sequence
        Returns:
            Tensor of shape (batch_size, count_L, d_model)
        """
        # Create position indices for each position in the sequence
        idx = torch.arange(1, count_L + 1, device=torch.device('cuda'))
        idx = idx.unsqueeze(0).unsqueeze(-1)  # Shape: (1, count_L, 1)
        
        # Create dimension indices for each dimension in d_model
        position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / self.d_model) for i in range(self.d_model)],
            device=torch.device('cuda'))
        position_vec = position_vec.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, d_model)
        
        # Compute positional encoding
        result = idx / position_vec
        
        # Apply sine to even indices and cosine to odd indices
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        
        # Expand to match batch size
        result = result.expand(self.num_batches, -1, -1)
        
        return result
    
    def forward(self, volume_matrix, non_pad_mask):
        """ Encode volume sequences via masked self-attention. """
        
        # Get sequence length and batch size from volume_matrix
        batch_size, count_L, _ = volume_matrix.size()
        self.num_batches = batch_size
        
        # Project volume data to d_model dimensions
        enc_output = self.volume_proj(volume_matrix)  # Shape: (batch_size, L, d_model)
        
        # prepare attention masks
        slf_attn_mask_subseq = get_subsequent_mask(volume_matrix)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=volume_matrix, seq_q=volume_matrix)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        
        pos_enc = self.positional_enc(count_L)
        
        for enc_layer in self.layer_stack:
            enc_output += pos_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        
        return enc_output
