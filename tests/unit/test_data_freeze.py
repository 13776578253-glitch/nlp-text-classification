from nlp_text_classification.data.contracts import Example
from nlp_text_classification.data.freeze import isolate_splits


def test_exclusions_preserve_test_and_split_order():
    train = [
        Example("t1", "val  text", (0,), "train"),
        Example("t2", "fit", (0,), "train"),
        Example("t3", "fit", (0,), "train"),
        Example("t4", "fit", (1,), "train"),
        Example("t5", "test text", (1,), "train"),
    ]
    validation = [
        Example("v1", "val text", (1,), "validation"),
        Example("v2", "test text", (0,), "validation"),
    ]
    test = [Example("e1", "test text", (1,), "test")]
    selected, excluded = isolate_splits({"train": train, "validation": validation, "test": test})
    assert selected == {"train": [train[1], train[3]], "validation": [validation[0]], "test": test}
    assert [item["sample_id"] for item in excluded] == ["t1", "t3", "t5", "v2"]
