import pytest

from nlp_text_classification.evaluation.batch_b_s2_report import _ranking, rare_gap
from nlp_text_classification.experiments.batch_b_s2 import budget_decision


@pytest.mark.parametrize(("new", "selected"), [
    (0.5, "s1_6_epochs"),
    (0.50999, "s1_6_epochs"),
    (0.51, "s2_12_epochs"),
])
def test_b1_budget_requires_predeclared_macro_gain(new, selected):
    assert budget_decision(0.5, new, 0.01, 1e-12)["selected_budget"] == selected


def test_rare_contribution_decomposes_the_signed_macro_gap():
    result = rare_gap([0.8, 0.4, 0.6], [0.2, 0.5, 0.7], [0])
    assert result["rare_contribution"] == pytest.approx(0.2)
    assert result["remaining_label_gap"] == pytest.approx(-0.1)
    assert result["all_label_gap"] == pytest.approx(
        result["rare_contribution"] + result["remaining_contribution"]
    )


def test_equal_subset_accuracy_is_reported_as_a_tied_rank():
    assert _ranking({"B1": 0.416, "B2": 0.416, "B3": 0.3874}) == [["B1", "B2"], ["B3"]]
