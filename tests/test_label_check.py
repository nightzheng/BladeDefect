from blade_defect.data.label_check import validate_seg_line


def test_valid_segmentation_line() -> None:
    assert validate_seg_line("0 0.1 0.1 0.5 0.1 0.5 0.5", num_classes=2) is None


def test_invalid_coordinate() -> None:
    assert "[0, 1]" in validate_seg_line("0 0.1 0.1 1.2 0.1 0.5 0.5")


def test_invalid_class() -> None:
    assert "类别范围" in validate_seg_line("2 0.1 0.1 0.5 0.1 0.5 0.5", num_classes=2)
