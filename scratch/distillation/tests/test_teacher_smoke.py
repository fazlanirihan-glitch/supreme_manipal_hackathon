"""Unit test for isolated Nexto teacher loading and forward inference."""
import hashlib
from pathlib import Path
import sys
import unittest
import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from collector import NextoTeacherHarness, NextoActionClassifier

CHECKPOINT_PATH = Path(r"c:\Users\Saheer Hawa\OneDrive\Desktop\NectoFamily-master\NectoFamily-master\nexto\nexto-model.pt")
EXPECTED_SHA = "bf5343b5eeacac6bf7cdb75dac4a5c14ba0f94d820eae75f00a211b6119d69fa"


class TestTeacherSmoke(unittest.TestCase):

    def test_checkpoint_exists_and_hash(self):
        """Teacher checkpoint must exist and match expected SHA-256."""
        self.assertTrue(CHECKPOINT_PATH.exists(), f"Missing teacher checkpoint at: {CHECKPOINT_PATH}")
        h = hashlib.sha256()
        with open(CHECKPOINT_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), EXPECTED_SHA)

    def test_teacher_load_and_forward(self):
        """Teacher must load via TorchScript on CPU and produce (1, 90) logits."""
        harness = NextoTeacherHarness(checkpoint_path=CHECKPOINT_PATH)
        ok, msg = harness.load_model()
        self.assertTrue(ok, f"Teacher loading failed: {msg}")

        q = np.zeros((1, 1, 32), dtype=np.float32)
        kv = np.zeros((1, 37, 24), dtype=np.float32)
        mask = np.zeros((1, 1, 37), dtype=np.float32)

        f_ok, logits, weights, f_msg = harness.run_forward_pass(q=q, kv=kv, mask=mask)
        self.assertTrue(f_ok, f"Forward pass failed: {f_msg}")
        self.assertEqual(logits.shape, (1, 90))
        self.assertTrue(np.all(np.isfinite(logits)))

        # Test action decode
        best_act = int(np.argmax(logits))
        ctrl = NextoActionClassifier.get_action_controls(best_act)
        self.assertEqual(len(ctrl), 8)
        self.assertTrue(np.all(np.isfinite(ctrl)))

    def test_environment_audit_flags_missing_rlgym(self):
        """Audit must accurately report missing rlgym_compat without failing."""
        env_info = NextoTeacherHarness.check_environment()
        self.assertTrue(env_info["torch_available"])
        self.assertTrue(env_info["rlbot_available"])
        self.assertFalse(env_info["rlgym_compat_available"])
        self.assertIn("No module named 'rlgym_compat'", env_info["rlgym_compat_error"])


if __name__ == "__main__":
    unittest.main()
