import torch
import torch.nn as nn

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
            self.decoder = nn.Linear(
                rnn_stack.n_hid, rnn_stack.emb_sz, bias=False)
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
        for c in self.children():
            if hasattr(c, 'reset'):
                c.reset()


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
