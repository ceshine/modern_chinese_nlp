import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import fastai.text
from fastai.core import BasicModel, to_gpu
from fastai.nlp import RNN_Learner
from fastai.lm_rnn import SequentialRNN

from .transformer_decoder import TransformerEncoder


class LanguageModelLoader:
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """

    MAX_PLUS = 5 * 5

    def __init__(self,
                 nums: np.array,
                 bs: int,
                 bptt: int,
                 backwards: bool = False,
                 batch_first: bool = False):
        self.bs, self.bptt, self.backwards = bs, bptt, backwards
        self.batch_first = batch_first
        self.data = self.batchify(nums)
        self.i, self.iter = 0, 0
        self.n = self.data.size(1) if self.batch_first else self.data.size(0)

    @property
    def max_possible_seq_len(self) -> int:
        return self.bptt + self.MAX_PLUS

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n - 1 and self.iter < len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random(
                ) < 0.95 else self.bptt / 2.
                seq_len = max(
                    5,
                    min(
                        int(np.random.normal(bptt, 5)),
                        self.max_possible_seq_len))
            if self.i + seq_len >= self.n:
                # ditch residuals
                break
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self):
        return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb * self.bs])
        data = data.reshape(self.bs, -1)
        if self.backwards:
            data = data[:, ::-1]
        if not self.batch_first:
            data = data.T
        return torch.from_numpy(data.astype("int64"))

    def get_batch(self, i, seq_len):
        source = self.data
        if self.batch_first:
            return (source[:, i:(i + seq_len)].contiguous(),
                    source[:, (i + 1):(i + 1 + seq_len)].contiguous().view(-1))
        else:
            return (source[i:(i + seq_len)].contiguous(),
                    source[(i + 1):(i + 1 + seq_len)].contiguous().view(-1))


class TransformerLanguageModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return m.blocks


class LanguageModelData(fastai.text.LanguageModelData):
    def get_transformer_model(self, opt_fn, emb_sz, max_seq_len, **kwargs):
        m = get_transformer_language_model(
            self.n_tok, max_seq_len, emb_sz, pad_token=self.pad_idx, **kwargs)
        model = TransformerLanguageModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)


class FlattenPredictions(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1, x.size(2))


def get_transformer_language_model(n_tok: int,
                                   max_seq_len: int,
                                   emb_sz: int,
                                   n_head: int,
                                   n_layer: int,
                                   pad_token: int,
                                   embd_pdrop: float = 0.1,
                                   attn_pdrop: float = 0.1,
                                   resid_pdrop: float = 0.1,
                                   afn: str = 'gelu'):
    enc = TransformerEncoder(
        vocab=n_tok,
        n_ctx=max_seq_len,
        n_embd=emb_sz,
        n_head=n_head,
        n_layer=n_layer,
        pad_token=pad_token,
        embd_pdrop=embd_pdrop,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        afn=afn)
    decoder = nn.Linear(emb_sz, n_tok, bias=False)
    decoder.weight = nn.Parameter(
        enc.embed.weight[:-max_seq_len])  # Tied weights
    return SequentialRNN(enc, decoder, FlattenPredictions())
