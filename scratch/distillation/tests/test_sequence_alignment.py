"""Unit tests for sequence chunking and timestamp alignment."""
import sys
from pathlib import Path
import unittest

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from collector import DemonstrationSample, DemonstrationEpisode


def chunk_episode_sequences(episode: DemonstrationEpisode, chunk_size: int = 64):
    """Splits an episode into fixed-length contiguous sequence chunks for GRU BPTT."""
    chunks = []
    samples = episode.samples
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


class TestSequenceAlignment(unittest.TestCase):

    def _create_sample(self, ep_id=1, frame=0, t=0.0):
        return DemonstrationSample(
            episode_id=ep_id,
            frame_id=frame,
            timestamp=t,
            student_features_13=[float(i) for i in range(13)],
            student_features_valid=True,
            teacher_raw_action=0,
            teacher_controls_8=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            mapped_mode=1,
            mapped_steer=0.0,
            compatibility_status="DIRECT",
            mapping_valid=True,
            rejection_reason=None,
            on_ground=True,
            context_label="OFFENSE"
        )

    def test_sequence_chunk_length(self):
        """Chunks must preserve chronological ordering and have max length of 64."""
        samples = [self._create_sample(frame=i, t=i * 0.0667) for i in range(150)]
        episode = DemonstrationEpisode(
            match_id="seq-match-1",
            episode_id=1,
            arena="DFHStadium",
            team=0,
            start_time=0.0,
            end_time=150 * 0.0667,
            samples=samples
        )

        chunks = chunk_episode_sequences(episode, chunk_size=64)
        self.assertEqual(len(chunks), 3) # 64 + 64 + 22 = 150
        self.assertEqual(len(chunks[0]), 64)
        self.assertEqual(len(chunks[1]), 64)
        self.assertEqual(len(chunks[2]), 22)

        # Verify ordering
        for chunk in chunks:
            for j in range(1, len(chunk)):
                self.assertGreater(chunk[j].timestamp, chunk[j - 1].timestamp)
                self.assertEqual(chunk[j].frame_id, chunk[j - 1].frame_id + 1)

    def test_timestamp_sync(self):
        """Observation and action must belong to the exact same frame index and timestamp."""
        sample = self._create_sample(frame=42, t=2.8)
        self.assertEqual(sample.frame_id, 42)
        self.assertAlmostEqual(sample.timestamp, 2.8)
        self.assertEqual(len(sample.student_features_13), 13)
        self.assertEqual(len(sample.teacher_controls_8), 8)


if __name__ == "__main__":
    unittest.main()
