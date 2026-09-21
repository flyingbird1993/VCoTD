from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from planning_reasoner.config import deep_merge, load_experiment_config


class ConfigTest(unittest.TestCase):
    def test_deep_merge_preserves_unmodified_nested_values(self) -> None:
        merged = deep_merge({"model": {"width": 32, "depth": 2}}, {"model": {"depth": 3}})
        self.assertEqual(merged, {"model": {"width": 32, "depth": 3}})

    def test_relative_base_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text("model:\n  width: 32\nseed: 7\n", encoding="utf-8")
            (root / "child.yaml").write_text(
                "base: base.yaml\nmodel:\n  depth: 2\n", encoding="utf-8"
            )
            loaded = load_experiment_config(root / "child.yaml")
        self.assertEqual(loaded["model"], {"width": 32, "depth": 2})
        self.assertEqual(loaded["seed"], 7)


if __name__ == "__main__":
    unittest.main()
