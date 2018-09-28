import torch
import pytest

from cnlp.transformer_decoder import Attention, dotdict


class TestAttention:
    @classmethod
    def setup_class(cls):
        cfg = dotdict({
            "n_head": 6,
            "attn_pdrop": 0,
            "resid_pdrop": 0.1,
            "afn": "gelu"
        })
        cls.attn = Attention(120, 100, cfg, scale=False)

    def test_dummy(self):
        assert isinstance(self.attn, Attention)

    def test_future_mask(self):
        w = torch.rand(32, 6, 50, 50)
        w_new = self.attn._future_blind_softmax(w)
        assert w_new.shape == (32, 6, 50, 50)
        assert w_new[0, 0, 0, 0].numpy() == pytest.approx(1)
        assert w_new[0, 0, 0, 1].numpy() == pytest.approx(0)
        assert w_new[0, 0, 0, 2].numpy() == pytest.approx(0)
        assert w_new[0, 0, 1, 2].numpy() == pytest.approx(0)
        assert w_new[0, 0, 2, 3].numpy() == pytest.approx(0)
        for i in range(50):
            assert w_new[0, 0, i, :(i + 1)].sum().numpy() == pytest.approx(1)

    def test_attn(self):
        q = torch.rand(32, 6, 50, 10)
        k = torch.rand(32, 6, 10, 50)
        v = torch.rand(32, 6, 50, 20)
        res = self.attn._attn(q, k, v)
        assert res.shape == (32, 6, 50, 20)
        # The first time stpe should the same as the value vector
        for i in range(20):
            assert res[0, 0, 0, i] == v[0, 0, 0, i]
            assert res[6, 5, 0, i] == v[6, 5, 0, i]

    def test_split_heads(self):
        x = torch.rand(32, 50, 120)
        x = self.attn.c_attn(x)
        assert x.shape == (32, 50, 360)
        query, key, value = x.split(self.attn.split_size, dim=2)
        query = self.attn.split_heads(query)
        assert query.shape == (32, 6, 50, 20)
        key = self.attn.split_heads(key, k=True)
        assert key.shape == (32, 6, 20, 50)
        value = self.attn.split_heads(value)
        assert value.shape == (32, 6, 50, 20)