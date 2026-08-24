from util.probe_cloud_host import classify


def test_platinum_8468_is_keep():
    result = classify("Intel(R) Xeon(R) Platinum 8468", 8)
    assert result["decision"] == "keep"
    assert result["cpu_class"] == "Platinum 8468"


def test_platinum_8352v_is_keep():
    result = classify("Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz", 8)
    assert result["decision"] == "keep"
    assert result["cpu_class"] == "Platinum 8352V"


def test_gold_6133_is_drop():
    result = classify("Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz", 8)
    assert result["decision"] == "drop"
    assert result["cpu_class"] == "Gold 6133"


def test_e5_2698_v4_is_drop():
    result = classify("Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz", 8)
    assert result["decision"] == "drop"
    assert result["cpu_class"] == "E5-2698 v4"


def test_unseen_model_is_unknown():
    result = classify("Intel(R) Xeon(R) Platinum 9999", 8)
    assert result["decision"] == "unknown"


def test_fewer_than_eight_cpus_is_drop():
    result = classify("Intel(R) Xeon(R) Platinum 8468", 4)
    assert result["decision"] == "drop"
