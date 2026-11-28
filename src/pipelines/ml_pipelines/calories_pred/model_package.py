"""Register trained models and build the BentoML prediction service."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from src.pipelines.ml_pipelines.calories_pred.model_register import (
    DEFAULT_REPORT_PATH,
    register_models_from_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BENTOML_HOME = PROJECT_ROOT / "Artifacts" / "data" / "ml_calories" / "bentoml"
BENTOFILE_PATH = (
    PROJECT_ROOT / "config" / "pipelines" / "ml_calories" / "bentofile.yaml"
)


def build_calories_service(
    report_path: str | Path = DEFAULT_REPORT_PATH,
    bentoml_home: str | Path = DEFAULT_BENTOML_HOME,
) -> None:
    """Register DVC model artifacts and build the service bundle."""
    bentoml_cli = shutil.which("bentoml")
    if bentoml_cli is None:
        raise FileNotFoundError(
            "BentoML CLI was not found; install the project ML requirements "
            "and activate that environment."
        )

    home = Path(bentoml_home).expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["BENTOML_HOME"] = str(home)

    previous_bentoml_home = os.environ.get("BENTOML_HOME")
    os.environ["BENTOML_HOME"] = str(home)
    try:
        register_models_from_report(report_path)
    finally:
        if previous_bentoml_home is None:
            os.environ.pop("BENTOML_HOME", None)
        else:
            os.environ["BENTOML_HOME"] = previous_bentoml_home

    subprocess.run(
        [
            bentoml_cli,
            "build",
            "--name",
            "calories-prediction",
            "--bentofile",
            str(BENTOFILE_PATH),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--bentoml-home", type=Path, default=DEFAULT_BENTOML_HOME)
    args = parser.parse_args()
    build_calories_service(args.report, args.bentoml_home)


if __name__ == "__main__":
    main()
