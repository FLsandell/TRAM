import pytest

from tram_genomics.io import load_dataset


def test_load_dataset_aligns_samples(tiny_data):
    features, labels = load_dataset(
        tiny_data["matrix"], tiny_data["groups"], "SP_CODE", "Group_A", "Group_B"
    )
    assert features.shape == (12, 4)
    assert labels.value_counts().to_dict() == {"Group_A": 6, "Group_B": 6}


def test_load_dataset_rejects_unknown_group(tiny_data):
    with pytest.raises(ValueError, match="No samples found"):
        load_dataset(tiny_data["matrix"], tiny_data["groups"], "SP_CODE", "Group_A", "Missing")
