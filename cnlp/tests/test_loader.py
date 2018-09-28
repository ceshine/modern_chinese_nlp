import torch
import pytest

import torch.nn as nn
import numpy as np

from cnlp.fastai_extended import LanguageModelLoader


@pytest.fixture(scope="module")
def loader():
    return LanguageModelLoader(
        np.random.randint(0, 50, 1000),
        target_length=30,
        bs=2,
        bptt=50,
        batch_first=True)


class TestLanguageModelLoder:
    def test_output_shape(self, loader):
        iterator = iter(loader)
        x, y = next(iterator)
        assert x.shape == (2, 50)
        assert y.shape == (2 * 30, )
        x, y = next(iterator)
        assert x.shape == (2, 50)
        assert y.shape == (2 * 30, )

    def test_x_and_y_consistency(self, loader):
        iterator = iter(loader)
        x, y = next(iterator)
        assert x[0, 21] == y[0]
        assert np.array_equal(x[0, -29:].numpy(), y[:29].numpy())
        assert np.array_equal(x[1, -29:].numpy(), y[30:59].numpy())
