# Stage B: Nexto Compatibility & Smoke Test Results
**Document ID:** `MW-SCRATCH-COMPAT-001`  
**Location:** `scratch/distillation/COMPATIBILITY_RESULTS.md`  
**Status:** Stage B Verification Complete  
**Classification Standard:** `[VERIFIED BY SOURCE]`, `[VERIFIED BY EXECUTION]`, `[EXPERIMENTAL RESULT]`, `[INFERRED]`, `[UNRESOLVED]`, `[REQUIRED APPROVAL]`

---

## 1. Executive Summary
This document records the empirical results of the Stage B implementation and smoke test for the isolated Nexto demonstration collector (`scratch/distillation/collector.py`).

All tests were executed strictly in the isolated `scratch/distillation/` directory. No production files, checkpoints, runtime PINs, or submission metadata were altered.

---

## 2. Environment Audit & Missing Dependency Reporting

Running `collector.py` executed a comprehensive local environment audit:

| Component | Status | Details | Verification Tag |
| :--- | :---: | :--- | :---: |
| **Python** | **PASS** | Python 3.12.7 (64-bit) | `[VERIFIED BY EXECUTION]` |
| **PyTorch** | **PASS** | `torch 2.14.1+cpu` | `[VERIFIED BY EXECUTION]` |
| **RLBot** | **PASS** | `rlbot 2.0.0-beta.55` | `[VERIFIED BY EXECUTION]` |
| **rlgym_compat** | **MISSING** | `No module named 'rlgym_compat'` | `[VERIFIED BY EXECUTION]` |

### Diagnostic Finding:
As required by Stage B Step 3, we explicitly report that **`rlgym_compat` is currently missing** from the local environment. Because `requirements.txt` in `NectoFamily-master` mandates `rlgym_compat @ git+https://github.com/JPK314/rlgym-compat` and `numpy==2.*`, installing it into the student virtual environment (`numpy==1.26.4`) would violate dependency isolation rules. Full live gameplay recording requires an isolated virtual environment `[VERIFIED BY SOURCE & EXECUTION]`.

---

## 3. Teacher Checkpoint & Model Inference Verification

The isolated teacher harness loaded and executed Nexto's TorchScript graph directly on CPU:

```text
Teacher Checkpoint: C:\Users\Saheer Hawa\OneDrive\Desktop\NectoFamily-master\NectoFamily-master\nexto\nexto-model.pt
File Size:          1,852,625 bytes
SHA-256 Hash:       bf5343b5eeacac6bf7cdb75dac4a5c14ba0f94d820eae75f00a211b6119d69fa
Model Type:         torch.jit.RecursiveScriptModule
Parameters:         444,032
Device:             CPU (torch.device("cpu"))
Forward Pass:       PASS (Input: (1,1,32), (1,37,24), (1,1,37) -> Output: (1, 90) logits)
```
`[VERIFIED BY EXECUTION]`

---

## 4. Nexto 90-Action Compatibility Audit

Running the automated action classifier on Nexto's authoritative 90-action lookup table produced the following distribution:

| Category | Action Count | Percentage | Student Mapping Semantics |
| :--- | :---: | :---: | :--- |
| **DIRECT** | 31 | 34.4% | Direct ground drive (Mode 1), neutral idle (Mode 0), single jump (Mode 2), straight front dodge (Mode 3). |
| **APPROXIMATED** | 27 | 30.0% | Forward boost drive $\to$ non-boost drive (Mode 1, max speed 1410 UU/s); diagonal dodge $\to$ front dodge + steer lock (Mode 3). |
| **UNREPRESENTABLE** | 32 | 35.6% | Reverse throttle, handbrake/powerslide, aerial pitch-up, boost flight. |

### Rejection Reasons Breakdown:
- `HANDBRAKE_UNAVAILABLE`: **25 actions** (student cannot execute powerslide cuts)
- `REVERSE_THROTTLE_UNAVAILABLE`: **6 actions** (student decoder has no negative throttle)
- `AERIAL_BOOST_FLIGHT_UNAVAILABLE`: **1 action** (student cannot pitch up or boost in mid-air)

`[VERIFIED BY EXECUTION: collector.py]`

---

## 5. Automated Unit Test Execution Suite

The test suite in `scratch/distillation/tests/` was executed using `unittest discover`:

```text
Ran 18 tests in 0.259s

OK
```

### Breakdown of Verified Test Cases:
1. `test_action_mapper.py` (8 tests):
   - `test_lookup_table_length`: Verified 90 discrete actions.
   - `test_direct_ground_drive`: Verified Mode 1 and continuous steer.
   - `test_direct_idle_coast`: Verified Mode 0.
   - `test_unrepresentable_reverse`: Verified rejection of negative throttle.
   - `test_unrepresentable_handbrake`: Verified rejection of powerslide.
   - `test_unrepresentable_aerial_pitch_up`: Verified rejection of pitch > 0 while jumping.
   - `test_approximated_boost_drive`: Verified Mode 1 approximation with speed flag.
   - `test_direct_front_dodge`: Verified Mode 3 Front Dodge.
2. `test_schema_validator.py` (5 tests):
   - `test_valid_sample`: Verified complete 13D sample structure.
   - `test_invalid_feature_dimension`: Verified rejection of vectors with $\neq 13$ features.
   - `test_nan_feature_rejection`: Verified rejection of NaN/Inf floating-point values.
   - `test_episode_timestamp_monotonicity`: Verified rejection of out-of-order frames.
   - `test_split_purity`: Verified 70/15/15 episode splitting with zero data leakage.
3. `test_sequence_alignment.py` (2 tests):
   - `test_sequence_chunk_length`: Verified 64-tick sequence chunking for GRU BPTT.
   - `test_timestamp_sync`: Verified synchronous observation-action frame alignment.
4. `test_teacher_smoke.py` (3 tests):
   - `test_checkpoint_exists_and_hash`: Verified exact SHA-256 of `nexto-model.pt`.
   - `test_teacher_load_and_forward`: Verified TorchScript graph load and (1, 90) output.
   - `test_environment_audit_flags_missing_rlgym`: Verified clean error trapping for missing `rlgym_compat`.

`[VERIFIED BY EXECUTION]`

---

## 6. Tested vs. Untested Components

### What Was Empirically Tested & Verified:
1. Teacher TorchScript loading and CPU forward inference.
2. Complete 90-action categorization and mathematical filtering rules.
3. Sample dataclass serialization and schema validation.
4. Feature dimension enforcement (exactly 13 scalar inputs).
5. Non-finite (NaN/Inf) data rejection.
6. Chronological 64-tick sequence chunking.
7. Whole-episode train/val/test splitting purity.

### What Remains Untested (Pending Stage C / External Environment):
1. **Live GamePacket Parsing by Nexto:** Nexto's `NextoObsBuilder` and `V1GameState` require `rlgym_compat`. Until `rlgym_compat` is installed in a separate isolated virtual environment, live matches cannot feed states into Nexto.
2. **Actual Gameplay Telemetry Recording:** No actual gameplay dataset has been recorded yet (Stage C).

---

## 7. Stage B Conclusion & Readiness Assessment
The Stage B collector harness (`scratch/distillation/collector.py`) is **complete, verified, and ready**. 

To collect live demonstrations in Stage C without corrupting the production environment, a dedicated isolated virtual environment (e.g. `scratch/distillation/.venv_collector`) should be created with `rlgym_compat` and `torch` `[REQUIRED APPROVAL]`. Candidate B remains securely sealed in `model/V13.pt`.
