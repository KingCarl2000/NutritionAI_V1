import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipelines.ml_pipelines.calories_pred.model_package import (
    BENTOFILE_PATH,
    build_calories_service,
)


class ModelPackageTests(unittest.TestCase):
    def test_registers_models_and_builds_with_dvc_store_as_home(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bentoml_home = Path(temp_dir) / "bentoml"
            previous_bentoml_home = os.environ.get("BENTOML_HOME")
            try:
                with (
                    patch(
                        "src.pipelines.ml_pipelines.calories_pred.model_package.shutil.which",
                        return_value="bentoml",
                    ),
                    patch(
                        "src.pipelines.ml_pipelines.calories_pred.model_package.register_models_from_report"
                    ) as register_models,
                    patch(
                        "src.pipelines.ml_pipelines.calories_pred.model_package.subprocess.run"
                    ) as run,
                ):
                    build_calories_service("training.json", bentoml_home)
            finally:
                if previous_bentoml_home is None:
                    os.environ.pop("BENTOML_HOME", None)
                else:
                    os.environ["BENTOML_HOME"] = previous_bentoml_home

            register_models.assert_called_once_with("training.json")
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["bentoml", "build", "--name"])
            self.assertIn(str(BENTOFILE_PATH), command)
            self.assertEqual(
                run.call_args.kwargs["env"]["BENTOML_HOME"], str(bentoml_home)
            )
            self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
