"""Isolated Nexto Demonstration Collector & Synchronized Feature Extractor.
Designed for Stage B Policy Distillation Research under Model Wars.

This module is strictly isolated from production files. It does NOT modify
model/V13.pt, src/runtime.py, src/policy.py, or any submission metadata.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# Paths
MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[1]
NEXTO_DIR = Path(r"c:\Users\Saheer Hawa\OneDrive\Desktop\NectoFamily-master\NectoFamily-master\nexto")
CONFIG_PATH = MODULE_DIR / "collector_config.json"


# ---------------------------------------------------------------------------
# 1. Nexto Action Lookup Table & Classifier
# ---------------------------------------------------------------------------

def make_nexto_lookup_table() -> np.ndarray:
    """Reconstructs the authoritative 90-action lookup table from Nexto agent.py."""
    actions = []
    # Ground actions
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                    )
    # Aerial actions
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append(
                            [boost, yaw, pitch, yaw, roll, jump, boost, handbrake]
                        )
    return np.array(actions, dtype=np.float32)


class NextoActionClassifier:
    """Classifies Nexto actions into student modes and compatibility categories."""

    LOOKUP_TABLE = make_nexto_lookup_table()

    # Status constants
    STATUS_DIRECT = "DIRECT"
    STATUS_APPROXIMATED = "APPROXIMATED"
    STATUS_UNREPRESENTABLE = "UNREPRESENTABLE"
    STATUS_UNCERTAIN = "UNCERTAIN"

    @classmethod
    def get_action_controls(cls, action_index: int) -> np.ndarray:
        if 0 <= action_index < len(cls.LOOKUP_TABLE):
            return cls.LOOKUP_TABLE[action_index]
        raise ValueError(f"Action index {action_index} out of range [0, {len(cls.LOOKUP_TABLE)-1}]")

    @classmethod
    def classify_action(cls, controls: np.ndarray) -> Tuple[int, float, str, bool, Optional[str]]:
        """Classifies an 8-channel Nexto control array into student targets.

        Controls: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

        Returns:
            mapped_mode (int): 0 (Neutral), 1 (Chase), 2 (Jump), 3 (Front Dodge)
            mapped_steer (float): Clamped continuous steer in [-1.0, 1.0]
            compatibility_status (str): DIRECT, APPROXIMATED, UNREPRESENTABLE, UNCERTAIN
            mapping_valid (bool): True if sample should be supervised
            rejection_reason (str or None): Explanation if unrepresentable
        """
        throttle = float(controls[0])
        steer = float(controls[1])
        pitch = float(controls[2])
        yaw = float(controls[3])
        roll = float(controls[4])
        jump = bool(controls[5])
        boost = bool(controls[6])
        handbrake = bool(controls[7])

        clamped_steer = float(np.clip(steer, -1.0, 1.0))

        # Check 1: Reverse throttle is strictly unrepresentable
        if throttle < 0.0:
            return 0, 0.0, cls.STATUS_UNREPRESENTABLE, False, "REVERSE_THROTTLE_UNAVAILABLE"

        # Check 2: Handbrake / powerslide turns are unrepresentable
        if handbrake:
            return 1, clamped_steer, cls.STATUS_UNREPRESENTABLE, False, "HANDBRAKE_UNAVAILABLE"

        # Check 3: Aerial pitch up is unrepresentable (Student pitch is <= 0.0)
        if jump and pitch > 0.0:
            return 2, clamped_steer, cls.STATUS_UNREPRESENTABLE, False, "AERIAL_PITCH_UP_UNAVAILABLE"

        # Check 4: Boost flight in mid-air
        if jump and boost and not (pitch == -1.0 and roll == 0.0 and yaw == 0.0):
            return 2, clamped_steer, cls.STATUS_UNREPRESENTABLE, False, "AERIAL_BOOST_FLIGHT_UNAVAILABLE"

        # Check 5: Pure aerial rotation without forward dodge
        if jump and (abs(yaw) > 0.0 or abs(roll) > 0.0) and pitch != -1.0:
            return 2, clamped_steer, cls.STATUS_UNREPRESENTABLE, False, "AERIAL_ROTATION_UNAVAILABLE"

        # Now classify representable actions:

        # Mode 3: Front Dodge
        if jump and pitch == -1.0:
            if abs(yaw) > 0.0 or abs(roll) > 0.0:
                # Diagonal dodge approximated by front flip + steer
                steer_target = -1.0 if yaw < 0 or roll < 0 else 1.0
                return 3, steer_target, cls.STATUS_APPROXIMATED, True, None
            return 3, clamped_steer, cls.STATUS_DIRECT, True, None

        # Mode 2: Single Jump Liftoff
        if jump:
            if boost:
                # Jump while boosting -> approximated to normal jump
                return 2, clamped_steer, cls.STATUS_APPROXIMATED, True, None
            return 2, clamped_steer, cls.STATUS_DIRECT, True, None

        # Mode 1: Forward Drive / Chase
        if throttle > 0.0:
            if boost:
                # Boost drive approximated to maximum non-boost forward drive
                return 1, clamped_steer, cls.STATUS_APPROXIMATED, True, None
            return 1, clamped_steer, cls.STATUS_DIRECT, True, None

        # Mode 0: Neutral / Coast
        if throttle == 0.0:
            return 0, 0.0, cls.STATUS_DIRECT, True, None

        return 0, 0.0, cls.STATUS_UNCERTAIN, False, "UNKNOWN_CONTROL_COMBINATION"


# ---------------------------------------------------------------------------
# 2. Data Containers & Synchronization Schemas
# ---------------------------------------------------------------------------

@dataclass
class DemonstrationSample:
    """A single synchronized timestep containing paired student and teacher data."""
    episode_id: int
    frame_id: int
    timestamp: float
    student_features_13: List[float]
    student_features_valid: bool
    teacher_raw_action: int
    teacher_controls_8: List[float]
    mapped_mode: int
    mapped_steer: float
    compatibility_status: str
    mapping_valid: bool
    rejection_reason: Optional[str]
    on_ground: bool
    context_label: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DemonstrationSample":
        return cls(**data)


@dataclass
class DemonstrationEpisode:
    """A contiguous episode of synchronized demonstration samples."""
    match_id: str
    episode_id: int
    arena: str
    team: int
    start_time: float
    end_time: float
    samples: List[DemonstrationSample]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "episode_id": self.episode_id,
            "arena": self.arena,
            "team": self.team,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sample_count": len(self.samples),
            "samples": [s.to_dict() for s in self.samples]
        }


# ---------------------------------------------------------------------------
# 3. Dataset Integrity Validator
# ---------------------------------------------------------------------------

class DemonstrationDatasetValidator:
    """Enforces schema consistency, feature dimension checks, and clean splitting."""

    @staticmethod
    def validate_sample(sample: DemonstrationSample) -> Tuple[bool, List[str]]:
        errors = []
        # Check feature length
        if len(sample.student_features_13) != 13:
            errors.append(f"Expected 13 student features, got {len(sample.student_features_13)}")
        # Check finite features
        if not all(math.isfinite(x) for x in sample.student_features_13):
            errors.append("Student features contain non-finite (NaN/Inf) values")
        # Check teacher controls
        if len(sample.teacher_controls_8) != 8:
            errors.append(f"Expected 8 teacher controls, got {len(sample.teacher_controls_8)}")
        if not all(math.isfinite(x) for x in sample.teacher_controls_8):
            errors.append("Teacher controls contain non-finite (NaN/Inf) values")
        # Check mode range
        if sample.mapped_mode not in (0, 1, 2, 3):
            errors.append(f"Mapped mode {sample.mapped_mode} not in [0, 1, 2, 3]")
        # Check steer range
        if not (-1.0 <= sample.mapped_steer <= 1.0):
            errors.append(f"Mapped steer {sample.mapped_steer} outside [-1.0, 1.0]")
        return len(errors) == 0, errors

    @staticmethod
    def validate_episode(episode: DemonstrationEpisode) -> Tuple[bool, List[str]]:
        errors = []
        if len(episode.samples) == 0:
            errors.append("Episode contains 0 samples")
            return False, errors

        prev_t = None
        for idx, s in enumerate(episode.samples):
            ok, sample_errs = DemonstrationDatasetValidator.validate_sample(s)
            if not ok:
                errors.extend([f"Frame {idx}: {e}" for e in sample_errs])
            if prev_t is not None and s.timestamp < prev_t:
                errors.append(f"Timestamp non-monotonic at frame {idx}: {s.timestamp} < {prev_t}")
            prev_t = s.timestamp

        return len(errors) == 0, errors

    @staticmethod
    def create_splits(episodes: List[DemonstrationEpisode],
                      train_ratio: float = 0.70,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      seed: int = 42) -> Dict[str, List[DemonstrationEpisode]]:
        """Splits datasets strictly by whole episodes to prevent adjacent-frame leakage."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        rng = random.Random(seed)
        shuffled = list(episodes)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_eps = shuffled[:n_train]
        val_eps = shuffled[n_train:n_train + n_val]
        test_eps = shuffled[n_train + n_val:]

        return {
            "train": train_eps,
            "validation": val_eps,
            "test": test_eps
        }

    @staticmethod
    def generate_manifest(splits: Dict[str, List[DemonstrationEpisode]]) -> Dict[str, Any]:
        manifest = {
            "format": "model_wars_distillation_dataset_v1",
            "schema_version": "1.0.0",
            "splits": {}
        }
        total_samples = 0
        total_valid = 0
        mode_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        rejection_counts = {}

        for split_name, eps in splits.items():
            split_samples = sum(len(e.samples) for e in eps)
            split_valid = sum(sum(1 for s in e.samples if s.mapping_valid) for e in eps)
            total_samples += split_samples
            total_valid += split_valid

            manifest["splits"][split_name] = {
                "episode_count": len(eps),
                "total_samples": split_samples,
                "valid_samples": split_valid,
                "episode_ids": [e.episode_id for e in eps]
            }

            for e in eps:
                for s in e.samples:
                    if s.mapping_valid:
                        mode_counts[s.mapped_mode] += 1
                    else:
                        r = s.rejection_reason or "UNKNOWN"
                        rejection_counts[r] = rejection_counts.get(r, 0) + 1

        manifest["total_episodes"] = sum(len(eps) for eps in splits.values())
        manifest["total_samples"] = total_samples
        manifest["total_valid_samples"] = total_valid
        manifest["mode_distribution"] = mode_counts
        manifest["rejection_distribution"] = rejection_counts
        return manifest


# ---------------------------------------------------------------------------
# 4. Teacher Harness & Diagnostics
# ---------------------------------------------------------------------------

class NextoTeacherHarness:
    """Manages isolated Nexto model loading and local environment diagnostics."""

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path or (NEXTO_DIR / "nexto-model.pt")
        self.model = None

    def load_model(self) -> Tuple[bool, str]:
        if not self.checkpoint_path.exists():
            return False, f"Checkpoint not found at: {self.checkpoint_path}"
        try:
            with open(self.checkpoint_path, "rb") as f:
                self.model = torch.jit.load(f, map_location="cpu")
            self.model.eval()
            return True, "Loaded TorchScript RecursiveScriptModule on CPU successfully"
        except Exception as e:
            return False, f"Failed to load TorchScript module: {e}"

    def run_forward_pass(self,
                          q: Optional[np.ndarray] = None,
                          kv: Optional[np.ndarray] = None,
                          mask: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """Runs an isolated forward pass on the loaded TorchScript graph."""
        if self.model is None:
            ok, msg = self.load_model()
            if not ok:
                return False, np.array([]), np.array([]), msg

        # Default shapes for 1v1
        if q is None:
            q = np.zeros((1, 1, 32), dtype=np.float32)
        if kv is None:
            kv = np.zeros((1, 37, 24), dtype=np.float32)
        if mask is None:
            mask = np.zeros((1, 1, 37), dtype=np.float32)

        try:
            t_q = torch.from_numpy(q).float()
            t_kv = torch.from_numpy(kv).float()
            t_mask = torch.from_numpy(mask).float()

            with torch.no_grad():
                out, weights = self.model((t_q, t_kv, t_mask))

            logits = out.cpu().numpy()
            return True, logits, weights, "Forward pass succeeded"
        except Exception as e:
            return False, np.array([]), np.array([]), f"Forward pass failed: {e}"

    @staticmethod
    def check_environment() -> Dict[str, Any]:
        """Audits the environment for RLBot, rlgym_compat, and PyTorch support."""
        results = {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "torch_available": False,
            "torch_version": None,
            "rlbot_available": False,
            "rlbot_version": None,
            "rlgym_compat_available": False,
            "rlgym_compat_error": None
        }

        try:
            import torch
            results["torch_available"] = True
            results["torch_version"] = torch.__version__
        except ImportError as e:
            results["torch_available"] = False

        try:
            import rlbot
            results["rlbot_available"] = True
            results["rlbot_version"] = getattr(rlbot, "__version__", "unknown")
        except ImportError:
            results["rlbot_available"] = False

        try:
            import rlgym_compat
            results["rlgym_compat_available"] = True
        except ImportError as e:
            results["rlgym_compat_available"] = False
            results["rlgym_compat_error"] = str(e)

        return results


# ---------------------------------------------------------------------------
# 5. CLI Execution & Diagnostic Tests
# ---------------------------------------------------------------------------

def run_smoke_test():
    print("=" * 60)
    print("STAGE B: NEXTO TEACHER SMOKE TEST & ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)

    # 1. Environment audit
    env_info = NextoTeacherHarness.check_environment()
    print("[1] Environment Audit:")
    print(f"    - Python: {env_info['python_executable']}")
    print(f"    - PyTorch: {env_info['torch_version']}")
    print(f"    - RLBot: {env_info['rlbot_version']}")
    print(f"    - rlgym_compat: {'INSTALLED' if env_info['rlgym_compat_available'] else 'MISSING (' + str(env_info['rlgym_compat_error']) + ')'}")

    # 2. Checkpoint check
    harness = NextoTeacherHarness()
    print("\n[2] Teacher Checkpoint Loading:")
    ok, msg = harness.load_model()
    print(f"    - Result: {'PASS' if ok else 'FAIL'}")
    print(f"    - Details: {msg}")

    # 3. Model forward pass
    if ok:
        print("\n[3] Model Forward Pass (Dummy 1v1 State):")
        f_ok, logits, weights, f_msg = harness.run_forward_pass()
        print(f"    - Result: {'PASS' if f_ok else 'FAIL'}")
        print(f"    - Details: {f_msg}")
        if f_ok:
            print(f"    - Output logits shape: {logits.shape} (Expected: (1, 90))")
            best_action = int(np.argmax(logits))
            controls = NextoActionClassifier.get_action_controls(best_action)
            mode, steer, status, valid, reason = NextoActionClassifier.classify_action(controls)
            print(f"    - Argmax Action: {best_action}")
            print(f"    - Decoded Controls: {controls.tolist()}")
            print(f"    - Mapped Student Mode: {mode}")
            print(f"    - Mapped Steer: {steer:.3f}")
            print(f"    - Compatibility Status: {status} (valid: {valid})")

    # 4. Action lookup table audit
    print("\n[4] Nexto 90-Action Compatibility Audit:")
    lookup = NextoActionClassifier.LOOKUP_TABLE
    status_counts = {}
    reasons = {}
    for i in range(len(lookup)):
        ctrl = lookup[i]
        _, _, stat, valid, reason = NextoActionClassifier.classify_action(ctrl)
        status_counts[stat] = status_counts.get(stat, 0) + 1
        if not valid:
            reasons[reason] = reasons.get(reason, 0) + 1

    print(f"    - Total Actions in Table: {len(lookup)}")
    for s, c in sorted(status_counts.items()):
        pct = (c / len(lookup)) * 100
        print(f"    - {s:18s}: {c:2d} ({pct:5.1f}%)")

    print("\n    Rejection Reasons Breakdown:")
    for r, c in sorted(reasons.items()):
        print(f"    - {r:32s}: {c:2d}")

    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
