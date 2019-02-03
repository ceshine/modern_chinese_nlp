"""DeepMoji Attention Layer

Adapted from https://github.com/huggingface/torchMoji/blob/master/torchmoji/attlayer.py
"""
import torch

import torch.nn as nn
from torch.nn.parameter import Parameter


class Attention(nn.Module):
    """Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size, return_attention=False, batch_first=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
            batch_first: if the first dimension is the batch number
        """
        super().__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        self.attention_vector.data.normal_(
            std=0.05)  # Initialize attention vector
        self.batch_first = batch_first

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.LongTensor):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Tensor): Tensor of input sequences with shape (seq_len, batch_size, chan)
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        # shape of logits: (batch_size, seq_len) or (seq_len, batch_size)
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        if self.batch_first:
            seq_dim = 1
            batch_dim = 0
        else:
            seq_dim = 0
            batch_dim = 1
        max_len = unnorm_ai.size(seq_dim)
        idxes = torch.arange(
            0, max_len, dtype=torch.long, device=inputs.device).unsqueeze(batch_dim)
        mask = (idxes < input_lengths.unsqueeze(seq_dim)).float()

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(
            dim=seq_dim, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        # shape: (batch_size, n_hid)
        representations = weighted.sum(dim=seq_dim)

        return (representations, attentions if self.return_attention else None)


class MultiAttention(nn.Module):
    """Combines Multiple Attention Modules"""

    def __init__(self, attention_size, heads=2, return_attention=False, batch_first=False):
        super().__init__()
        self.attention_size = attention_size
        self.return_attention = return_attention
        self.heads = heads
        self.attns = nn.ModuleList([
            Attention(attention_size, return_attention=return_attention,
                      batch_first=batch_first)
            for i in range(heads)])

    def __repr__(self):
        s = '{name}({attention_size}, {heads}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.LongTensor):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Tensor): Tensor of input sequences with shape (seq_len, batch_size, chan)
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        contexts, attn_weights = [], []
        for attn in self.attns:
            tmp = attn(inputs, input_lengths)
            contexts.append(tmp[0]), attn_weights.append(tmp[1])
        return (torch.cat(contexts, dim=1), attn_weights if self.return_attention else None)
