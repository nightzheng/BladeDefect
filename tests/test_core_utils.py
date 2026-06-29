from pathlib import Path

import pytest
import yaml

from blade_defect.utils import device as device_module
from blade_defect.utils.device import resolve_device
from blade_defect.utils.files import load_dataset_config, resolved_data_yaml, write_data_yaml
from blade_defect.utils.paths import posix_path, resolve_config_paths, resolve_path


def test_auto_device_uses_first_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "_cuda_is_available", lambda: True)
    assert resolve_device("auto") == 0


def test_auto_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "_cuda_is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


@pytest.mark.parametrize(("value", "expected"), [(0, 0), ("0", 0), ("cpu", "cpu")])
def test_explicit_device(value: str | int, expected: str | int) -> None:
    assert resolve_device(value) == expected


def test_relative_path_resolves_against_base(tmp_path: Path) -> None:
    assert resolve_path("datasets/images", tmp_path) == (tmp_path / "datasets" / "images").resolve()


def test_relative_windows_and_linux_separators_are_accepted(tmp_path: Path) -> None:
    linux_style = resolve_path("datasets/images/train", tmp_path)
    windows_style = resolve_path(r"datasets\images\train", tmp_path)
    assert linux_style == windows_style


def test_absolute_path_is_not_rebased(tmp_path: Path) -> None:
    absolute = (tmp_path / "data").resolve()
    assert resolve_path(absolute, tmp_path / "other") == absolute


def test_project_root_controls_config_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "train.yaml"
    config_path.parent.mkdir()
    config_path.touch()
    config, root = resolve_config_paths(
        {"project_root": "..", "data": "configs/data.yaml", "project": "runs/train"},
        config_path,
        ("data", "project"),
    )
    assert root == tmp_path.resolve()
    assert config["data"] == (tmp_path / "configs" / "data.yaml").resolve()
    assert config["project"] == (tmp_path / "runs" / "train").resolve()


def test_windows_absolute_path_serializes_without_backslashes() -> None:
    assert "\\" not in posix_path(r"C:\datasets\blade")


def test_data_yaml_generation_uses_portable_paths(tmp_path: Path) -> None:
    dataset = tmp_path / "datasets" / "blade"
    dataset.mkdir(parents=True)
    output = tmp_path / "configs" / "data.yaml"

    write_data_yaml(dataset, output, names={0: "crack"})

    text = output.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert "\\" not in text
    assert payload["path"] == "../datasets/blade"
    assert payload["train"] == "images/train"
    assert load_dataset_config(output)["path"] == dataset.resolve()


def test_runtime_data_yaml_has_absolute_dataset_root(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = tmp_path / "data.yaml"
    source.write_text("path: dataset\ntrain: images/train\nval: images/val\n", encoding="utf-8")

    with resolved_data_yaml(source) as normalized:
        payload = yaml.safe_load(Path(normalized).read_text(encoding="utf-8"))
        assert payload["path"] == dataset.resolve().as_posix()

    assert not Path(normalized).exists()
