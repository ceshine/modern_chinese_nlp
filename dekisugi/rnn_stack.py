import warnings

import torch
import torch.nn as nn

from .rnn_reg import WeightDrop, LockedDropout


def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    return h.detach() if type(h) == torch.Tensor else tuple(repackage_var(v) for v in h)


class RNNStack(nn.Module):
    """ A stack of LSTM or QRNN layers to drive the network, and
        variational dropouts in the embedding and LSTM/QRNN layers

        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    def __init__(self, emb_sz: int, n_hid: int, n_layers: int, bidir=False,
                 dropouth=0.3, dropouti=0.65, wdrop=0.5, unit_type="lstm", qrnn=False):
        """Default constructor for the RNNStack class

        Parameters
        ----------
        emb_sz : int
            the embedding size used to encode each token
        n_hid : int
            number of hidden units per layer.
        n_layers : int
            number of layers to use in the architecture.
        bidir : bool, optional
            Use bidirectional layout. (only used when qrnn=False)
        dropouth : float, optional
            dropout to apply to the activations going from one layer to another.
        dropouti : float, optional
            dropout to apply to the input layer.
        wdrop : float, optional
            dropout used for a LSTM's internal (or hidden) recurrent weights.
            (only used when qrnn=False)
        qrnn : bool, optional
            use QRNN instead of LSTM.
        """
        super().__init__()
        print("[RNNStack] *qrnn* is deprecated, use unit_type=\"qrnn\" instead.")
        unit_type = unit_type.strip().lower()
        self.qrnn = (unit_type == "qrnn") or qrnn
        self.unit = nn.LSTM if unit_type == "lstm" else nn.GRU
        self.bs = 1
        self.ndir = 2 if bidir else 1
        assert not (
            self.qrnn and self.bidir), "QRNN does not support bidirectionality."
        if self.qrnn:
            # Using QRNN requires cupy: https://github.com/cupy/cupy
            from .torchqrnn.qrnn import QRNNLayer
            self.rnns = [QRNNLayer(emb_sz if l == 0 else n_hid, n_hid,
                                   save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(n_layers)]
            if wdrop:
                for rnn in self.rnns:
                    rnn.linear = WeightDrop(
                        rnn.linear, wdrop, weights=['weight'])
        else:
            self.rnns = [self.unit(emb_sz if l == 0 else n_hid, n_hid // self.ndir,
                                   1, bidirectional=bidir) for l in range(n_layers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.emb_sz, self.n_hid, self.n_layers = emb_sz, n_hid, n_layers
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList(
            [LockedDropout(dropouth) for l in range(n_layers)])

    def forward(self, emb):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        sl, bs, emb_sz = emb.size()
        assert emb_sz == self.emb_sz, "input size does not match model size"
        if bs != self.bs:
            self.bs = bs
            self.reset()
        with torch.set_grad_enabled(self.training):
            raw_output = self.dropouti(emb)
            new_hidden, raw_outputs = [], []
            for l, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.n_layers - 1:
                    raw_output = drop(raw_output)
            self.hidden = repackage_var(new_hidden)
        return raw_outputs, self.hidden

    def one_hidden(self, l):
        nh = self.n_hid // self.ndir
        return next(self.parameters()).new_empty(self.ndir, self.bs, nh).zero_()

    def reset(self):
        if self.qrnn or (self.unit is nn.GRU):
            [r.reset() for r in self.rnns]
            self.hidden = [self.one_hidden(l) for l in range(self.n_layers)]
        else:
            self.hidden = [(self.one_hidden(l), self.one_hidden(l))
                           for l in range(self.n_layers)]
