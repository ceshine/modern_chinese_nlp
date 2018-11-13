"""Generic Sequence Models (excl. seq2seq)"""
from typing import Sequence, List
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from helperbot.bot import BaseBot

from .rnn_stack import RNNStack
from .embeddings import BasicEmbeddings


class LinearBlock(nn.Module):
    """Simple Linear Block with Dropout and BatchNorm

    Adapted from fast.ai v0.7
    """

    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)
        nn.init.kaiming_normal_(self.lin.weight)
        nn.init.constant_(self.lin.bias, 0)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))


class PoolingFCN(nn.Module):
    """FCN that make use of all sequence inputs.

    Average pooling + max pooling + last time step.

    Adapted from fast.ai v0.7
    """

    def __init__(self, layers: List[int], drops: Sequence[float]):
        super().__init__()
        assert len(layers) == len(drops) + 1
        layers[0] = layers[0] * 3
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(bs, -1)

    def forward(self, input_tensors):
        output = input_tensors[-1]
        _, bs, _ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x


class SequenceModel(nn.Module):
    def __init__(self, embeddings: BasicEmbeddings, encoder: RNNStack, fcn: nn.Module):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.fcn = fcn

    def forward(self, input_tokens: torch.LongTensor):
        # input_tokens shape (seq_length, batch_size)
        embeddings = self.embeddings(input_tokens)
        # embeddings shape (seq_length, batch_size, emb_sz)
        # Remember to reset the hidden states!
        self.encoder.reset()
        rnn_output, _ = self.encoder(embeddings)
        # rnn_output shape (seq_length, batch_size, n_hid)
        outputs = self.fcn(rnn_output)
        # outputs shape (seq_length, batch_size, voc_size)
        return outputs

    def get_layer_groups(self):
        return [self.embeddings, *self.encoder.rnns, self.fcn]


def get_sequence_model(
        voc_sz: int,
        emb_sz: int,
        pad_idx: int,
        dropoute: float,
        rnn_hid: int,
        rnn_layers: int,
        bidir: bool,
        dropouth: float,
        dropouti: float,
        wdrop: float,
        qrnn: float,
        fcn_layers: Sequence[int],
        fcn_dropouts: Sequence[float]):
    embeddings = BasicEmbeddings(voc_sz, emb_sz, pad_idx, dropoute)
    rnn_stack = RNNStack(emb_sz, rnn_hid, rnn_layers, bidir,
                         dropouth, dropouti, wdrop, qrnn)
    fcn = PoolingFCN([rnn_hid] + list(fcn_layers), fcn_dropouts)
    return SequenceModel(embeddings, rnn_stack, fcn)


class SequenceRegressorBot(BaseBot):
    name = "seq_regressor"

    def __init__(self, model, train_loader, val_loader, *, optimizer, clip_grad=0,
                 avg_window=2000, log_dir="./data/cache/logs/", log_level=logging.INFO,
                 checkpoint_dir="./data/cache/model_cache/", batch_idx=0, echo=False,
                 device="cuda:0", use_tensorboard=False):
        super().__init__(
            model, train_loader, val_loader,
            optimizer=optimizer,
            clip_grad=clip_grad,
            avg_window=avg_window,
            log_dir=log_dir,
            log_level=log_level,
            checkpoint_dir=checkpoint_dir,
            batch_idx=batch_idx,
            echo=echo,
            device=device,
            use_tensorboard=use_tensorboard
        )
        self.criterion = torch.nn.MSELoss()
        self.loss_format = "%.6f"

    @staticmethod
    def extract_prediction(output):
        return output

    def load_encoder(self, source_dir: str = None, prefix: str = ""):
        if source_dir is None:
            source_path = self.checkpoint_dir
        else:
            source_path = Path(source_dir)
        self.model.embeddings.load_state_dict(torch.load(
            source_path / f"{prefix}embeddings.pth").state_dict())
        self.model.encoder.load_state_dict(torch.load(
            source_path / f"{prefix}rnn_stack.pth").state_dict())
