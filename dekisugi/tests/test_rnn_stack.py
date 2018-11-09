import torch
import pytest

from dekisugi.rnn_stack import RNNStack


@pytest.fixture(scope="module")
def lstm_model() -> RNNStack:
    return RNNStack(
        emb_sz=10,
        n_hid=16,
        n_layers=3,
        bidir=False,
        dropouth=0.5,
        dropouti=0.5,
        wdrop=0.1,
        qrnn=False
    )


class TestRNNStack:
    def test_one_hidden(self, lstm_model: RNNStack):
        for l in (0, 2):
            tensor = lstm_model.one_hidden(l)
            assert tensor.shape == (1, 1, 16)
            assert tensor.equal(torch.zeros(1, 1, 16, dtype=torch.float32))

    def test_lstm_reset(self, lstm_model: RNNStack):
        lstm_model.hidden = [
            (torch.rand(1, 1, 16, dtype=torch.float32),
             torch.rand(1, 1, 16, dtype=torch.float32))
            for _ in range(lstm_model.n_layers)]
        for i in range(lstm_model.n_layers):
            assert not lstm_model.hidden[i][0].equal(
                torch.zeros(1, 1, 16, dtype=torch.float32))
            assert not lstm_model.hidden[i][1].equal(
                torch.zeros(1, 1, 16, dtype=torch.float32))
        lstm_model.reset()
        for i in range(lstm_model.n_layers):
            assert lstm_model.hidden[i][0].equal(
                torch.zeros(1, 1, 16, dtype=torch.float32))
            assert lstm_model.hidden[i][1].equal(
                torch.zeros(1, 1, 16, dtype=torch.float32))

    def test_lstm_forward(self, lstm_model: RNNStack):
        input_tensor = torch.rand(5, 2, 10, dtype=torch.float32)
        lstm_model.reset()
        output, hidden = lstm_model(input_tensor)
        # check output shapes
        assert len(output) == 3
        assert output[-1].size() == (5, 2, 16)
        # check if hidden states have been changed
        for i in range(lstm_model.n_layers):
            assert not lstm_model.hidden[i][0].equal(
                torch.zeros(1, 1, 16, dtype=torch.float32))
            assert not lstm_model.hidden[i][1].equal(
                torch.zeros(1, 1, 16, dtype=torch.float32))
