"""Tests for config path resolution from different working directories and env var overrides."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest


class TestConfigPathResolution(unittest.TestCase):
    """Paths should resolve relative to PROJECT_ROOT, not CWD."""

    def test_project_root_is_repo_root(self):
        from src.config import PROJECT_ROOT
        # PROJECT_ROOT should be an absolute path pointing at the repo.
        self.assertTrue(PROJECT_ROOT.is_absolute())
        self.assertTrue((PROJECT_ROOT / "config.yaml").exists())

    def test_env_var_overrides_project_root(self):
        # PROJECT_ROOT env var should override the derived root.
        old = os.environ.get("PROJECT_ROOT")
        try:
            with tempfile.TemporaryDirectory() as td:
                fake_root = Path(td) / "fake_project"
                fake_root.mkdir()
                (fake_root / "config.yaml").write_text("api:\n  server_port: 9999\n")
                os.environ["PROJECT_ROOT"] = str(fake_root)
                # Force reimport of config module
                if "src.config" in sys.modules:
                    del sys.modules["src.config"]
                from src.config import PROJECT_ROOT
                self.assertEqual(PROJECT_ROOT, fake_root.resolve())
        finally:
            if old is None:
                os.environ.pop("PROJECT_ROOT", None)
            else:
                os.environ["PROJECT_ROOT"] = old

    def test_data_char_root_env_override(self):
        from src.config import SETTINGS
        old = os.environ.get("DATA_CHAR_ROOT")
        try:
            with tempfile.TemporaryDirectory() as td:
                custom = Path(td) / "custom_char_data"
                custom.mkdir()
                os.environ["DATA_CHAR_ROOT"] = str(custom)
                if "src.config" in sys.modules:
                    del sys.modules["src.config"]
                from src.config import SETTINGS as S2
                self.assertEqual(S2.data.char_dataset_dir, custom.resolve())
        finally:
            if old is None:
                os.environ.pop("DATA_CHAR_ROOT", None)
            else:
                os.environ["DATA_CHAR_ROOT"] = old

    def test_cnn_checkpoint_env_override(self):
        from src.config import SETTINGS
        old = os.environ.get("CNN_CHECKPOINT")
        try:
            with tempfile.TemporaryDirectory() as td:
                custom_ckpt = Path(td) / "custom.pth"
                custom_ckpt.write_text("not a real checkpoint")
                os.environ["CNN_CHECKPOINT"] = str(custom_ckpt)
                if "src.config" in sys.modules:
                    del sys.modules["src.config"]
                from src.config import SETTINGS as S2
                self.assertEqual(S2.models.cnn_checkpoint, custom_ckpt.resolve())
        finally:
            if old is None:
                os.environ.pop("CNN_CHECKPOINT", None)
            else:
                os.environ["CNN_CHECKPOINT"] = old


if __name__ == "__main__":
    unittest.main()