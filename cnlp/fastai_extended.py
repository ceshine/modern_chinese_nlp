import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import fastai.text
from fastai.core import BasicModel, to_gpu
from fastai.nlp import RNN_Learner
from fastai.lm_rnn import SequentialRNN, LinearBlock
# from fastai.dataloader import DataLoader

from .transformer_decoder import TransformerEncoder


class LanguageModelLoader:
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """

    MAX_PLUS = 25

    def __init__(self,
                 nums: np.array,
                 bs: int,
                 bptt: int,
                 target_length: int,
                 backwards: bool = False,
                 batch_first: bool = False,
                 randomize_bptt: bool = False):
        self.bs, self.bptt, self.backwards = bs, bptt, backwards
        self.batch_first = batch_first
        self.data = self.batchify(nums)
        self.i, self.iter = 0, 0
        self.n = self.data.size(1) if self.batch_first else self.data.size(0)
        self.randomize_bptt = randomize_bptt
        self.target_length = target_length

    @property
    def max_possible_seq_len(self) -> int:
        if self.randomize_bptt is False:
            return self.bptt
        return self.bptt + self.MAX_PLUS

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n - 1 and self.iter < len(self):
            if self.randomize_bptt:
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
            else:
                seq_len = self.bptt
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
        target_offset = max(0, seq_len - self.target_length)
        if self.batch_first:
            return (source[:, i:(i + seq_len)].contiguous(),
                    source[:, (i + 1 + target_offset):(
                        i + 1 + seq_len)].contiguous().view(-1))
        else:
            return (source[i:(i + seq_len)].contiguous(),
                    source[(i + 1 + target_offset):(
                        i + 1 + seq_len)].contiguous().view(-1))


class TransformerLanguageModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return m.blocks


class LanguageModelData(fastai.text.LanguageModelData):
    def get_transformer_model(self, opt_fn, emb_sz, max_seq_len, **kwargs):
        m = get_transformer_language_model(
            self.n_tok,
            max_seq_len,
            self.trn_dl.target_length,
            emb_sz,
            pad_token=self.pad_idx,
            **kwargs)
        model = TransformerLanguageModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)


class FlattenPredictions(nn.Module):
    def __init__(self, target_len: int):
        super().__init__()
        self.target_len = target_len

    def forward(self, x):
        return x[:, -self.target_len:, :].contiguous().view(-1, x.size(2))


def get_transformer_language_model(n_tok: int,
                                   max_seq_len: int,
                                   target_length: int,
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
    return SequentialRNN(enc, decoder, FlattenPredictions(target_length))


class PoolingLinearClassifier(nn.Module):
    def __init__(self, layers, drops, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i])
            for i in range(len(layers) - 1)
        ])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        if self.batch_first:
            return f(x.permute(0, 2, 1), (1, )).view(bs, -1)
        return f(x.permute(1, 2, 0), (1, )).view(bs, -1)

    def forward(self, output):
        if self.batch_first:
            sl, bs, _ = output.size()
        else:
            bs, sl, _ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x


def get_transformer_classifier(n_tok: int,
                               max_seq_len: int,
                               emb_sz: int,
                               n_head: int,
                               n_layer: int,
                               clf_layers: int,
                               pad_token: int,
                               embd_pdrop: float = 0.1,
                               attn_pdrop: float = 0.1,
                               resid_pdrop: float = 0.1,
                               clf_pdrop: float = 0.1,
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
    classifier = PoolingLinearClassifier(clf_layers, clf_pdrop)
    return SequentialRNN(enc, classifier)
