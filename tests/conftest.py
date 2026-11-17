import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("run")


@pytest.fixture(scope="session")
def sample_csv(workdir):
    path = workdir / "sample.csv"
    generate(rows=3000, seed=1).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def config_path(workdir, sample_csv):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["data"]["path"] = str(sample_csv)
    cfg["data"]["features"] = []  # synthetic data lacks some real columns
    cfg["output_dir"] = str(workdir / "models")
    cfg["mlflow"]["registered_model_name"] = None
    path = workdir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture(scope="session")
def trained(workdir, config_path, monkeypatch_session):
    from iot_ids.config import load_config
    from iot_ids.train import run

    monkeypatch_session.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(workdir / 'mlflow.db').as_posix()}")
    monkeypatch_session.chdir(workdir)  # keep MLflow artifacts out of the project folder
    cfg = load_config(config_path)
    result = run(cfg, config_path=str(config_path))
    return cfg, result


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()
