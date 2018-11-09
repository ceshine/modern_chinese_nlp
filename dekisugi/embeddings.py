import torch.nn as nn
import torch

from .rnn_reg import EmbeddingDropout


class BasicEmbeddings(nn.Module):
    """A simple wrapper around an embeddings matrix
       that comes with optional embedding dropouts
    """
    initrange = 0.1

    def __init__(self, voc_sz: int, emb_sz: int, pad_idx: int, dropoute: float = 0):
        """Default constructor for the BasicEmbeddings class

        Parameters
        ----------
        voc_sz : int
            number of vocabulary (or tokens) in the source dataset.
        emb_sz : int
            the embedding size used to encode each token.
        pad_idx : int
            the int value used for padding text.
        dropoute : float, optional
            dropout to apply to the embedding layer. (the default is 0)
        """
        super().__init__()
        self.voc_sz, self.emb_sz, self.dropoute = voc_sz, emb_sz, dropoute
        self.encoder = nn.Embedding(voc_sz, emb_sz, padding_idx=pad_idx)
        if dropoute > 0:
            self.encoder = EmbeddingDropout(self.encoder, dropoute)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

    def forward(self, input_tensor: torch.LongTensor):
        """ Invoked during the forward propagation of the BasicEmbeddings module.

        Parameters
        ----------
        input_tensor: torch.Tensor
            A Long Tensor with shape (seq_length, batch_size)
        """
        return self.encoder(input_tensor)
