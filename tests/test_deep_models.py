import pytest
torch = pytest.importorskip("torch")
from src.deep_models import LSTMSentimentClassifier, MultiKernelCNNClassifier, AttentionPoolingClassifier


@pytest.mark.parametrize("model_cls", [
    LSTMSentimentClassifier, MultiKernelCNNClassifier, AttentionPoolingClassifier
])
def test_forward_shape(model_cls):
    model = model_cls(vocab_size=50, num_classes=4)
    ids = torch.randint(1, 50, (3, 12))
    lengths = torch.tensor([12, 9, 5])
    logits = model(ids, lengths)
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()


def test_cnn_handles_sequences_shorter_than_kernel():
    model = MultiKernelCNNClassifier(vocab_size=20, num_classes=3, kernel_sizes=(3, 4, 5))
    ids = torch.tensor([[1, 2]])
    assert model(ids).shape == (1, 3)
