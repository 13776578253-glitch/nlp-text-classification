import pytest

torch = pytest.importorskip("torch")

from nlp_text_classification.models.neural import RecurrentClassifier


@pytest.mark.parametrize("architecture", ["vanilla_rnn", "lstm", "bilstm"])
def test_single_layer_dropout_changes_training_outputs_only(architecture):
    torch.manual_seed(42)
    model = RecurrentClassifier(architecture, 12, 5, 8, 2, 0.7)
    tokens = torch.tensor([[2, 3, 4], [5, 6, 0]])
    lengths = torch.tensor([3, 2])
    model.train()
    assert not torch.equal(model(tokens, lengths), model(tokens, lengths))
    model.eval()
    torch.testing.assert_close(model(tokens, lengths), model(tokens, lengths))


@pytest.mark.parametrize("architecture", ["vanilla_rnn", "lstm", "bilstm"])
def test_trailing_padding_does_not_change_valid_representation(architecture):
    torch.manual_seed(42)
    model = RecurrentClassifier(architecture, 12, 5, 8, 2, 0.3).eval()
    tokens = torch.tensor([[2, 3, 4], [5, 6, 0]])
    # Alter the padded embedding values and append additional trailing entries.
    padded = torch.tensor([[2, 3, 4, 7, 8], [5, 6, 9, 8, 7]])
    lengths = torch.tensor([3, 2])
    torch.testing.assert_close(model.representation(tokens, lengths),
                               model.representation(padded, lengths))


def test_bilstm_uses_last_forward_and_first_backward_outputs_in_order():
    torch.manual_seed(42)
    model = RecurrentClassifier("bilstm", 12, 5, 4, 2, 0).eval()
    tokens = torch.tensor([[2, 3, 4], [5, 6, 0]])
    lengths = torch.tensor([3, 2])
    expected = []
    for row, length in zip(tokens, lengths, strict=True):
        embedded = model.embedding(row[:length].unsqueeze(0))
        output, _ = model.encoder(embedded)
        expected.append(torch.cat((output[0, -1, :4], output[0, 0, 4:])))
    torch.testing.assert_close(model.representation(tokens, lengths), torch.stack(expected))
