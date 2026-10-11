"""Unit tests for DemonstrationDatasetValidator in scratch/distillation/collector.py."""
import sys
from pathlib import Path
import unittest

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from collector import (
    DemonstrationSample,
    DemonstrationEpisode,
    DemonstrationDatasetValidator
)


class TestSchemaValidator(unittest.TestCase):

    def _create_sample(self, ep_id=1, frame=0, t=0.0, valid=True, features=None):
        return DemonstrationSample(
            episode_id=ep_id,
            frame_id=frame,
            timestamp=t,
            student_features_13=features or [0.1] * 13,
            student_features_valid=True,
            teacher_raw_action=0,
            teacher_controls_8=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            mapped_mode=1,
            mapped_steer=0.0,
            compatibility_status="DIRECT" if valid else "UNREPRESENTABLE",
            mapping_valid=valid,
            rejection_reason=None if valid else "REVERSE_THROTTLE_UNAVAILABLE",
            on_ground=True,
            context_label="OFFENSE"
        )

    def test_valid_sample(self):
        sample = self._create_sample()
        ok, errors = DemonstrationDatasetValidator.validate_sample(sample)
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)

    def test_invalid_feature_dimension(self):
        """Feature vectors of length other than 13 must fail validation."""
        sample = self._create_sample(features=[0.1] * 12)
        ok, errors = DemonstrationDatasetValidator.validate_sample(sample)
        self.assertFalse(ok)
        self.assertTrue(any("Expected 13 student features" in e for e in errors))

    def test_nan_feature_rejection(self):
        """Samples containing NaN or Inf features must fail validation."""
        features = [0.1] * 12 + [float("nan")]
        sample = self._create_sample(features=features)
        ok, errors = DemonstrationDatasetValidator.validate_sample(sample)
        self.assertFalse(ok)
        self.assertTrue(any("non-finite" in e for e in errors))

    def test_episode_timestamp_monotonicity(self):
        """Episodes with non-monotonic timestamps must fail validation."""
        samples = [
            self._create_sample(frame=0, t=1.0),
            self._create_sample(frame=1, t=0.5), # Decreasing timestamp
        ]
        episode = DemonstrationEpisode(
            match_id="test-match-1",
            episode_id=1,
            arena="DFHStadium",
            team=0,
            start_time=0.0,
            end_time=1.0,
            samples=samples
        )
        ok, errors = DemonstrationDatasetValidator.validate_episode(episode)
        self.assertFalse(ok)
        self.assertTrue(any("non-monotonic" in e for e in errors))

    def test_split_purity(self):
        """Dataset splits must be partitioned strictly by whole episode IDs without leakage."""
        episodes = [
            DemonstrationEpisode(
                match_id=f"match-{i}",
                episode_id=i,
                arena="DFHStadium",
                team=i % 2,
                start_time=0.0,
                end_time=10.0,
                samples=[self._create_sample(ep_id=i, frame=0, t=0.0)]
            )
            for i in range(20)
        ]

        splits = DemonstrationDatasetValidator.create_splits(
            episodes, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )

        train_ids = set(e.episode_id for e in splits["train"])
        val_ids = set(e.episode_id for e in splits["validation"])
        test_ids = set(e.episode_id for e in splits["test"])

        # Check total count
        self.assertEqual(len(train_ids) + len(val_ids) + len(test_ids), 20)
        # Check pairwise disjoint sets (no overlap / no leakage)
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))


if __name__ == "__main__":
    unittest.main()
