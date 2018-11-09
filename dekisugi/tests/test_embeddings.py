import torch
import pytest

from dekisugi.embeddings import BasicEmbeddings


@pytest.fixture(scope="module")
def small_embeddings() -> BasicEmbeddings:
    return BasicEmbeddings(100, 10, 0, 0.5)


class TestBasicEmbeddings:
    def test_size(self, small_embeddings: BasicEmbeddings):
        assert small_embeddings.encoder.weight.size() == (100, 10)

    def test_forward(self, small_embeddings: BasicEmbeddings):
        input_tensor = torch.randint(0, 100, (5, 2), dtype=torch.int64)
        output_tensor = small_embeddings(input_tensor)
        assert output_tensor.size() == (5, 2, 10)
        # Check dropout; should have a very low chance of failing
        flag = False
        for _ in range(20):
            output_tensor = small_embeddings(
                torch.randint(0, 100, (1, 1), dtype=torch.int64))
            if output_tensor[0, 0].equal(torch.zeros(10, dtype=torch.float32)):
                flag = True
                break
        assert flag is True, "embedding dropout seems to be broken."
