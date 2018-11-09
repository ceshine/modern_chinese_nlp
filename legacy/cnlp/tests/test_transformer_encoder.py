import torch
import pytest

import torch.nn as nn

from cnlp.transformer_decoder import TransformerEncoder
from cnlp.fastai_extended import get_transformer_language_model

TARGET_LENGTH = 90


@pytest.fixture(scope="module")
def model():
    return get_transformer_language_model(
        n_tok=1000,
        max_seq_len=100,
        target_length=TARGET_LENGTH,
        emb_sz=120,
        n_head=6,
        n_layer=10,
        pad_token=2,
        embd_pdrop=0,
        attn_pdrop=0,
        resid_pdrop=0)


class TestTransformerEncoder:
    def test_tied_weight(self, model):
        assert isinstance(model[0].embed, nn.Embedding)
        assert model[0].embed.weight.shape == (1000 + 100, 120)
        assert model[1].weight.shape == (1000, 120)
        assert model[1].weight[0, 1] != 0
        model[0].embed.weight[0, 1] = 0
        assert model[1].weight[0, 1] == 0

    def test_output_shapes(self, model):
        x = (torch.rand(32, TARGET_LENGTH) * 1000).long()
        res = model(x)
        assert res.shape == (32 * TARGET_LENGTH, 1000)
        res = model[0](x)
        assert res.shape == (32, TARGET_LENGTH, 120)
        res = model[1](res)
        assert res.shape == (32, TARGET_LENGTH, 1000)

    def test_flatten(self, model):
        x = torch.rand(32, TARGET_LENGTH, 120)
        res = model[2](x)
        assert x[0, 0, 0] == res[0, 0]
        assert x[0, 1, 0] == res[1, 0]
        assert x[1, 0, 0] == res[TARGET_LENGTH, 0]