from datahouse.frame_corpus import split_rows


def test_split_is_exact_disjoint_and_input_order_independent():
    rows = [{"fingerprint": f"{i:064x}"} for i in range(10_000)]
    a = split_rows(rows)
    b = split_rows(list(reversed(rows)))
    assert [row["fingerprint"] for row in a] == [row["fingerprint"] for row in b]
    assert len(a) == 1_000
    assert sum(row["split"] == "train" for row in a) == 800
    assert sum(row["split"] == "validation" for row in a) == 100
    assert sum(row["split"] == "test" for row in a) == 100
