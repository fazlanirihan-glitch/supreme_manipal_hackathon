"""Unit tests for NextoActionClassifier in scratch/distillation/collector.py."""
import sys
from pathlib import Path
import unittest
import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from collector import NextoActionClassifier


class TestActionMapper(unittest.TestCase):

    def test_lookup_table_length(self):
        """Lookup table must contain exactly 90 discrete actions."""
        table = NextoActionClassifier.LOOKUP_TABLE
        self.assertEqual(len(table), 90)
        self.assertEqual(table.shape, (90, 8))

    def test_direct_ground_drive(self):
        """Forward throttle with steer and no boost should map to Mode 1 (Chase)."""
        # [throttle=1, steer=-1, pitch=0, yaw=0, roll=0, jump=0, boost=0, handbrake=0]
        controls = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mode, steer, status, valid, reason = NextoActionClassifier.classify_action(controls)
        self.assertEqual(mode, 1)
        self.assertAlmostEqual(steer, -1.0)
        self.assertEqual(status, NextoActionClassifier.STATUS_DIRECT)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_direct_idle_coast(self):
        """Zero throttle without boost or jump should map to Mode 0 (Neutral)."""
        controls = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mode, steer, status, valid, reason = NextoActionClassifier.classify_action(controls)
        self.assertEqual(mode, 0)
        self.assertAlmostEqual(steer, 0.0)
        self.assertEqual(status, NextoActionClassifier.STATUS_DIRECT)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_unrepresentable_reverse(self):
        """Negative throttle must be flagged UNREPRESENTABLE."""
        controls = np.array([-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mode, steer, status, valid, reason = NextoActionClassifier.classify_action(controls)
        self.assertEqual(status, NextoActionClassifier.STATUS_UNREPRESENTABLE)
        self.assertFalse(valid)
        self.assertEqual(reason, "REVERSE_THROTTLE_UNAVAILABLE")

    def test_unrepresentable_handbrake(self):
        """Handbrake / powerslide must be flagged UNREPRESENTABLE."""
        controls = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        mode, steer, status, valid, reason = NextoActionClassifier.classify_action(controls)
        self.assertEqual(status, NextoActionClassifier.STATUS_UNREPRESENTABLE)
        self.assertFalse(valid)
        self.assertEqual(reason, "HANDBRAKE_UNAVAILABLE")

    def test_unrepresentable_aerial_pitch_up(self):
        """Aerial pitch up (pitch > 0 while jumping) must be flagged UNREPRESENTABLE."""
        controls = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        mode, steer, status, valid, reason = NextoActionClassifier.classify_action(controls)
        self.assertEqual(status, NextoActionClassifier.STATUS_UNREPRESENTABLE)
        self.assertFalse(valid)
        self.assertEqual(reason, "AERIAL_PITCH_UP_UNAVAILABLE")

    def test_approximated_boost_drive(self):
        """Boost drive should be marked APPROXIMATED to Mode 1 (Chase)."""
        controls = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        mode, steer, status, valid, reason = NextoActionClassifier.classify_action(controls)
        self.assertEqual(mode, 1)
        self.assertAlmostEqual(steer, 0.5)
        self.assertEqual(status, NextoActionClassifier.STATUS_APPROXIMATED)
        self.assertTrue(valid)

    def test_direct_front_dodge(self):
        """Jump with pitch=-1 and no yaw/roll should map to Mode 3 (Front Dodge)."""
        controls = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        mode, steer, status, valid, reason = NextoActionClassifier.classify_action(controls)
        self.assertEqual(mode, 3)
        self.assertEqual(status, NextoActionClassifier.STATUS_DIRECT)
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
