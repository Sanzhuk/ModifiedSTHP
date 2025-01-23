import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
import Utils

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size() # [1, 4000]

    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

def get_sparse_mask(seq, window_sizes):
    sz_b, len_s = seq.size()
    
    sparse_mask = torch.tril(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8))
    
    # Create a matrix with indices corresponding to the sequence length
    
    row_indices = torch.transpose(torch.arange(len_s, device=seq.device).unsqueeze(0).expand(len_s, -1), 0, 1)
    col_indices = torch.transpose(torch.arange(len_s, device=seq.device).unsqueeze(1).expand(-1, len_s), 0, 1)
    
    # Compute valid regions based on window_sizes
    valid_region = (col_indices < row_indices + window_sizes.unsqueeze(1))
    
    # Mask out the valid regions
    sparse_mask[valid_region] = 0
    
    # Expand the sparse mask for all batches
    sparse_mask = sparse_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return sparse_mask
    
class Encoder_event(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, lambda_window):
        super().__init__()

        self.d_model = d_model
        self.lambda_window = lambda_window
        
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
    
        return result * non_pad_mask # [1, 4000, 512]

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        window_sizes = Utils.get_window_sizes(event_time[0], lambda_window=self.lambda_window)
        
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_sparse = get_sparse_mask(event_time, window_sizes)
        
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask_sparse = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq + slf_attn_mask_sparse).gt(0)

        # print("SLF ATTN MASK EVENT")
        # print(slf_attn_mask.shape)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)
        
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output

