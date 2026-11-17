from nlp_text_classification.data.loaders.movie_reviews_zh import audit_dataset, load_split


def test_audit_counts_blanks_and_text_overlap_without_rewriting_inputs(tmp_path):
    train = "0\t相同  文本\n\n0\t相同 文本\n"
    (tmp_path / "train.txt").write_text(train, encoding="utf-8")
    (tmp_path / "validation.txt").write_text("1\t相同 文本\n", encoding="utf-8")
    (tmp_path / "test.txt").write_text("1\t其他 文本\n", encoding="utf-8")
    (tmp_path / "wiki_word2vec_50.bin").write_bytes(b"2 50\n")

    audit = audit_dataset(tmp_path)
    assert audit["counts"] == {"train": 2, "validation": 1, "test": 1}
    assert audit["blank_rows"]["train"] == 1
    assert audit["within_split"]["train"]["extra_duplicate_rows"] == 1
    assert audit["cross_split"]["train:validation"] == {
        "shared_texts": 1,
        "texts_with_conflicting_label_sets": 1,
    }
    assert audit["cross_split"]["train:test"]["shared_texts"] == 0
    assert audit["word2vec"]["dimension"] == 50
    rows, _ = load_split(tmp_path, "train")
    assert [row.sample_id for row in rows] == ["train:1", "train:3"]
    assert rows[0].text == "相同  文本"
    assert (tmp_path / "train.txt").read_text(encoding="utf-8") == train
