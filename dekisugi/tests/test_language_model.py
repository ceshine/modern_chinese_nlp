import torch
import pytest

from dekisugi.language_model import get_language_model, RNNLanguageModel


@pytest.fixture(scope="module")
def small_tied_weights() -> RNNLanguageModel:
    return get_language_model(
        voc_sz=100,
        emb_sz=10,
        pad_idx=0,
        dropoute=0.1,
        rnn_hid=16,
        rnn_layers=3,
        bidir=False,
        dropouth=0.1,
        dropouti=0.1,
        wdrop=0.1,
        qrnn=False,
        tie_weights=True
    )


class TestRNNLanguageModel:
    def test_forward(self, small_tied_weights: RNNLanguageModel):
        input_tensor = torch.randint(0, 100, (5, 2), dtype=torch.int64)
        logits, states = small_tied_weights(input_tensor)
        assert logits.size() == (5, 2, 100)
        assert small_tied_weights.decoder[-1].weight is small_tied_weights.embeddings.encoder.weight
        assert len(states) == 3
        assert len(states[-1]) == 2
        assert states[-1][0].size() == (1, 2, 16)
        assert states[-1][1].size() == (1, 2, 16)
