from pathlib import Path

from blade_defect.evaluation import AblationRunner


def test_ablation_runner(tmp_path: Path) -> None:
    config = tmp_path / "ablation.yaml"
    config.write_text(
        "base:\n  epochs: 1\nexperiments:\n  - name: baseline\n    overrides:\n      imgsz: 320\n",
        encoding="utf-8",
    )
    runner = AblationRunner(config, tmp_path / "output")
    records = runner.run(lambda name, params: {"map50": 0.5})
    assert records[0]["name"] == "baseline"
    assert records[0]["imgsz"] == 320
    assert (tmp_path / "output" / "summary.csv").exists()
