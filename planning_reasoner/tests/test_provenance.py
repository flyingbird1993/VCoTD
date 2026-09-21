from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from planning_reasoner.metrics.provenance import (
    build_manifest,
    record_artifacts,
    sha256_file,
    write_manifest,
)


class ProvenanceTest(unittest.TestCase):
    def test_hash_and_manifest_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("stable\n", encoding="utf-8")
            self.assertEqual(
                sha256_file(source), "2b92ea252be0fbc26f70317cdaa7b6411ea634b50d55338cd8c495e4dbf25d1d"
            )
            manifest = build_manifest({"name": "test"}, 7, root, {"source": source})
            path = write_manifest(root / "run", manifest)
            artifact = root / "run" / "prediction.json"
            artifact.write_text("{}\n", encoding="utf-8")
            record_artifacts(root / "run", {"prediction": artifact})
            self.assertTrue(path.is_file())
            self.assertEqual(manifest["seed"], 7)
            self.assertIn("artifacts", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
