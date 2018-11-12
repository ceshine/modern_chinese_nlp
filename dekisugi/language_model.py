import logging

import torch
import torch.nn as nn
import numpy as np

from helperbot.bot import BaseBot

from .rnn_stack import RNNStack
from .embeddings import BasicEmbeddings


class RNNLanguageModel(nn.Module):
    def __init__(self, embeddings: BasicEmbeddings, rnn_stack: RNNStack, tie_weights: bool = True):
        super().__init__()
        self.embeddings = embeddings
        self.rnn_stack = rnn_stack
        self.tie_weights = tie_weights
        if self.rnn_stack.emb_sz != self.rnn_stack.n_hid:
            self.decoder = nn.Sequential(
                nn.Linear(rnn_stack.n_hid, embeddings.emb_sz),
                nn.Linear(embeddings.emb_sz, embeddings.voc_sz, bias=False)
            )
            self.init_fcn(self.decoder)
            if tie_weights:
                self.decoder[-1].weight = self.embeddings.encoder.weight
        else:
            self.decoder = nn.Sequential(
                nn.Linear(rnn_stack.n_hid, embeddings.voc_sz, bias=False))
            self.init_fcn(self.decoder)
            if tie_weights:
                self.decoder.weight = self.embeddings.encoder.weight

    @staticmethod
    def init_fcn(fcn):
        fcn = [fcn] if not isinstance(fcn, nn.Sequential) else fcn
        for m in fcn:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_layer_groups(self):
        if self.tie_weights:
            return [*self.rnn_stack.rnns, self.decoder]
        return [self.embeddings, *self.rnn_stack.rnns, self.decoder]

    def forward(self, input_tokens: torch.LongTensor):
        # input_tokens shape (seq_length, batch_size)
        embeddings = self.embeddings(input_tokens)
        # embeddings shape (seq_length, batch_size, emb_sz)
        rnn_output, rnn_states = self.rnn_stack(embeddings)
        # rnn_output shape (seq_length, batch_size, n_hid)
        logits = self.decoder(rnn_output[-1])
        # logits shape (seq_length, batch_size, voc_size)
        return logits, rnn_states

    def reset(self):
        for ch in self.children():
            if hasattr(ch, 'reset'):
                ch.reset()


def get_language_model(
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
        tie_weights: bool = True):
    embeddings = BasicEmbeddings(voc_sz, emb_sz, pad_idx, dropoute)
    rnn_stack = RNNStack(emb_sz, rnn_hid, rnn_layers, bidir,
                         dropouth, dropouti, wdrop, qrnn)
    return RNNLanguageModel(embeddings, rnn_stack, tie_weights)


class LanguageModelLoader:
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)

    (Directly copied from fast.ai v0.7 with minor variable changes.)
    The first batch returned is always bptt+25; the max possible width.
    This is done because of they way that pytorch allocates cuda memory in order to prevent multiple
    buffers from being created as the batch width grows.
    """

    def __init__(self, arr, batch_size: int, bptt: int, backwards: bool = False, randomize: bool = False):
        """Constructor for LanguageModelLoader

        Parameters
        ----------
        arr : np.array
            An numpy array containing all the tokens.
        batch_size : int
            The desired batch size.
        bptt : int
            The target sequence length (roughly) of the batch
        backwards : bool, optional
            Flip the order of the sequence if True. (the default is False)
        randomize : bool, optional
            Randomize the sequence length.
        """
        self.batch_size, self.bptt, self.backwards = batch_size, bptt, backwards
        self.randomize = randomize
        self.data = self.batchify(arr)
        self.i, self.iter = 0, 0
        self.n_tokens = len(self.data)

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n_tokens-1 and self.iter < len(self):
            if self.randomize:
                if self.i == 0:
                    seq_len = self.bptt + 5 * 5
                else:
                    bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                    seq_len = max(5, int(np.random.normal(bptt, 5)))
            else:
                seq_len = self.bptt
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self):
        return self.n_tokens // self.bptt - 1

    def batchify(self, data):
        n_batches = data.shape[0] // self.batch_size
        data = np.array(data[:n_batches*self.batch_size])
        data = data.reshape(self.batch_size, -1).T
        if self.backwards:
            data = data[::-1]
        return torch.from_numpy(data).long()

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].contiguous().view(-1)


class LMBot(BaseBot):
    name = "LM"

    def __init__(self, model, train_loader, val_loader, *, optimizer, clip_grad=0,
                 avg_window=2000, log_dir="./data/cache/logs/", log_level=logging.INFO,
                 checkpoint_dir="./data/cache/model_cache/", batch_idx=0, echo=False,
                 device="cuda:0", use_tensorboard=False):
        super().__init__(
            model, train_loader, val_loader,
            optimizer=optimizer,
            clip_grad=0,
            avg_window=avg_window,
            log_dir=log_dir,
            log_level=log_level,
            checkpoint_dir=checkpoint_dir,
            batch_idx=batch_idx,
            echo=echo,
            device=device,
            use_tensorboard=use_tensorboard
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_format = "%.4f"

    @staticmethod
    def extract_prediction(output):
        return output[0].view(-1, output[0].size(2))

    def eval(self, loader):
        self.model.reset()
        loss = super().eval(loader)
        self.model.reset()
        return loss

    def export_encoder(self, target_path: str, prefix: str = ""):
        """Export the embedding and RNN stack

        WARNING: It also loads the model from the target path

        Parameters
        ----------
        target_path : str
            Where the dumped state dict is stored.
        prefix : str
            The prefix used in the file names.
        """
        self.load_model(target_path)
        torch.save(
            self.model.embeddings,
            self.checkpoint_dir / f"{prefix}embeddings.pth")
        torch.save(
            self.model.rnn_stack,
            self.checkpoint_dir / f"{prefix}rnn_stack.pth")
