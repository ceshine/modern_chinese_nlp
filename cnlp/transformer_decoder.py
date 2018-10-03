"""Transformer decoder as in the OpenAI paper

Adapted from huggingface/pytorch-openai-transformer-lm.
(https://github.com/huggingface/pytorch-openai-transformer-lm)
"""
import copy
import json
import math
import re
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {'relu': nn.ReLU, 'swish': swish, 'gelu': gelu}


def dropout_mask(x, sz, dropout):
    """ Applies a dropout mask whose size is determined by passed argument 'sz'.
    Args:
        x (nn.Variable): A torch Variable object
        sz (tuple(int, int, int)): The expected size of the new tensor
        dropout (float): The dropout fraction to apply

    This method uses the bernoulli distribution to decide which activations to keep.
    Additionally, the sampled activations is rescaled is using the factor 1/(1 - dropout).

    In the example given below, one can see that approximately .8 fraction of the
    returned tensors are zero. Rescaling with the factor 1/(1 - 0.8) returns a tensor
    with 5's in the unit places.

    The official link to the pytorch bernoulli function is here:
        http://pytorch.org/docs/master/torch.html#torch.bernoulli

    Examples:
        >>> a_Var = torch.autograd.Variable(torch.Tensor(2, 3, 4).uniform_(0, 1), requires_grad=False)
        >>> a_Var
            Variable containing:
            (0 ,.,.) =
              0.6890  0.5412  0.4303  0.8918
              0.3871  0.7944  0.0791  0.5979
              0.4575  0.7036  0.6186  0.7217
            (1 ,.,.) =
              0.8354  0.1690  0.1734  0.8099
              0.6002  0.2602  0.7907  0.4446
              0.5877  0.7464  0.4257  0.3386
            [torch.FloatTensor of size 2x3x4]
        >>> a_mask = dropout_mask(a_Var.data, (1,a_Var.size(1),a_Var.size(2)), dropout=0.8)
        >>> a_mask
            (0 ,.,.) =
              0  5  0  0
              0  0  0  5
              5  0  5  0
            [torch.FloatTensor of size 1x3x4]
    """
    return x.new(*sz).bernoulli_(1 - dropout) / (1 - dropout)


class LockedDropout(nn.Module):
    def __init__(self, p=0.5, dim=1):
        super().__init__()
        self.p = p
        assert dim in (1, 2)
        self.dim = dim

    def forward(self, x):
        if not self.training or not self.p: return x
        with torch.set_grad_enabled(False):
            if self.dim == 1:
                mask = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
            else:
                mask = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return mask * x


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


# class Conv1D(nn.Module):
#     def __init__(self, nf, rf, nx):
#         super(Conv1D, self).__init__()
#         self.rf = rf
#         self.nf = nf
#         if rf == 1:  # faster 1x1 conv
#             w = torch.empty(nx, nf)
#             nn.init.normal_(w, std=0.02)
#             self.w = Parameter(w)
#             self.b = Parameter(torch.zeros(nf))
#         else:  # was used to train LM
#             raise NotImplementedError

#     def forward(self, x):
#         if self.rf == 1:
#             size_out = x.size()[:-1] + (self.nf, )
#             x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
#             x = x.view(*size_out)
#         else:
#             raise NotImplementedError
#         return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.register_buffer(
            'b',
            torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = nn.Linear(nx, n_state * 3)  # Conv1D(n_state * 3, 1, nx)
        self.c_proj = nn.Linear(nx, n_state)  # Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = LockedDropout(cfg.resid_pdrop, dim=1)

        nn.init.kaiming_normal_(self.c_attn.weight)
        nn.init.kaiming_normal_(self.c_proj.weight)
        nn.init.constant_(self.c_attn.bias, 0)
        nn.init.constant_(self.c_proj.bias, 0)

    def _future_blind_softmax(self, w):
        # TF implem method: mask_attn_weights
        mask = self.b[:, :, :w.size(2), :w.size(2)]
        w = w.masked_fill(mask == 0, -1e9)
        return nn.Softmax(dim=-1)(w)

    def _attn(self, q, k, v):
        # w shape: batch_size, n_head, seq_len, seq_len
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = self._future_blind_softmax(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1), )
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = nn.Linear(nx, n_state)  # Conv1D(n_state, 1, nx)
        self.c_proj = nn.Linear(n_state, nx)  # Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

        nn.init.kaiming_normal_(self.c_fc.weight)
        nn.init.kaiming_normal_(self.c_proj.weight)
        nn.init.constant_(self.c_fc.bias, 0)
        nn.init.constant_(self.c_proj.bias, 0)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class TransformerEncoder(nn.Module):
    """Implement the "Transformer Encoder" described in the OpenAI paper
    """

    def __init__(self,
                 vocab: int,
                 n_ctx: int,
                 pad_token: int,
                 n_embd: int,
                 n_head: int,
                 n_layer: int,
                 embd_pdrop: float,
                 attn_pdrop: float,
                 resid_pdrop: float,
                 afn: str = "gelu"):
        """
        Note: vocab should only include regular vocabulary and special characters.
        """
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab + n_ctx, n_embd, padding_idx=pad_token)
        self.drop = LockedDropout(embd_pdrop, dim=1)
        self.n_ctx = n_ctx
        cfg = dotdict({
            "n_head": n_head,
            "n_embd": n_embd,
            "attn_pdrop": attn_pdrop,
            "resid_pdrop": resid_pdrop,
            "afn": afn
        })
        block = Block(n_ctx, cfg, scale=True)
        self.blocks = [copy.deepcopy(block) for _ in range(n_layer)]
        self.blocks = nn.ModuleList(self.blocks)
        nn.init.normal_(self.embed.weight, std=0.02)

    def _prepare_input_tensor(self, x):
        n_batch = x.size(0)
        seq_len = x.size(1)
        x_w_context = torch.zeros((n_batch, seq_len, 2)).long().to(x.device)
        x_w_context[:, :, 0] = x
        x_w_context[:, :, 1] = torch.arange(self.vocab + self.n_ctx - seq_len,
                                            self.vocab + self.n_ctx).long().to(
                                                x.device)
        return x_w_context

    def forward(self, x):
        # x = x.view(-1, x.size(-2), x.size(-1))
        x = self._prepare_input_tensor(x)
        with torch.set_grad_enabled(self.training):
            e = self.embed(x)
            # Add the position information to the input embeddings
            h = self.drop(e.sum(dim=2))
            # # L2 norm
            # h = F.normalize(h, p=2, dim=-1)
            for block in self.blocks:
                h = block(h)
        return h


class LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg):
        super(LMHead, self).__init__()
        self.n_embd = cfg.n_embd
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight  # Tied weights

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(h_trunc)
        return lm_logits


class MultipleChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, clf_token, cfg):
        super(MultipleChoiceHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        # To reproduce the noise_shape parameter of TF implementation
        self.dropout = nn.Dropout2d(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, 1)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        # Classification logits
        clf_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = clf_h.view(-1, x.size(1), self.n_embd, 1)
        # This double transposition is there to replicate the behavior
        # of the noise_shape argument in the tensorflow
        # implementation.  For more details, see
        # https://github.com/huggingface/pytorch-openai-transformer-lm/issues/11
        clf_h = self.dropout(clf_h.transpose(1, 2)).transpose(1, 2)
        clf_h = clf_h.contiguous().view(-1, self.n_embd)
        clf_logits = self.linear(clf_h)

        return clf_logits.view(-1, x.size(1))


class ClfHead(nn.Module):
    """Classification Head for the transformer

    TODO: test this class."""

    def __init__(self, clf_token, cfg, n_class):
        super(ClfHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, n_class)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        clf_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)

        return clf_logits


class SimilarityHead(nn.Module):
    """ Similarity Head for the transformer

        TODO: test this class."""

    def __init__(self, clf_token, cfg):
        super(SimilarityHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, 1)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        sim_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        sim_h = sim_h[flat == self.clf_token, :]
        sim_h = self.dropout(sim_h)
        sim_h = sim_h.sum(dim=1)
        sim_logits = self.linear(sim_h)

        return sim_logits


# class DoubleHeadModel(nn.Module):
#     """ Transformer with language model and task specific heads """

#     def __init__(self, cfg, clf_token, task_head_type, vocab=40990, n_ctx=512):
#         super(DoubleHeadModel, self).__init__()
#         self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
#         self.lm_head = LMHead(self.transformer, cfg)
#         if isinstance(task_head_type, str):
#             if task_head_type == 'multiple_choice':
#                 self.task_head = MultipleChoiceHead(clf_token, cfg)
#             elif task_head_type == 'similarity':
#                 self.task_head = SimilarityHead(clf_token, cfg)
#             elif task_head_type == 'inference':
#                 # the three classes correspond to entailment, contradiction and neutral.
#                 self.task_head = ClfHead(clf_token, cfg, 3)
#             else:
#                 raise ValueError(
#                     "task_head_type is expected to be 'multiple_choice' "
#                     "'similarity', 'inference' or ('classification', n_class) "
#                     f"got {task_head_type}.")
#         elif isinstance(task_head_type, collections.abc.Sequence) and len(task_head_type) == 2 and \
#                 task_head_type[0] == 'classification':
#             n_class = task_head_type[1]
#             self.task_head = ClfHead(clf_token, cfg, n_class)
#         else:
#             raise ValueError(
#                 "task_head_type is expected to be 'multiple_choice' "
#                 "'similarity', 'inference' or ('classification', n_class) "
#                 f"got {task_head_type}.")

#     def forward(self, x):
#         h = self.transformer(x)
#         lm_logits = self.lm_head(h)
#         task_logits = self.task_head(h, x)

#         return lm_logits, task_logits


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_CONFIG = dotdict({
    'n_embd': 768,
    'n_head': 12,
    'n_layer': 12,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'afn': 'gelu',
    'clf_pdrop': 0.1
})