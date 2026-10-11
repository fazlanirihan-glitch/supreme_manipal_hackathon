# Nexto-to-Student Demonstration Dataset Specification & Schema
**Document ID:** `MW-SPEC-DATA-001`  
**Schema Version:** `1.0.0`  
**Target:** Policy Distillation Data Collection Harness  
**Classification Standard:** `[VERIFIED BY SOURCE]`, `[VERIFIED BY EXECUTION]`, `[EXPERIMENTAL RESULT]`, `[INFERRED]`, `[UNRESOLVED]`, `[REQUIRED APPROVAL]`

---

## 1. Scope and Objective
This document defines the storage schema, synchronization protocol, and validation requirements for the demonstration dataset used to train the Model Wars Student policy. 

To preserve the competition integrity rules:
1. **Zero Runtime Feature Injection:** Metadata, teacher logits, and game-state ground truths are stored **only for offline training loss supervision**. The deployed student input interface remains strictly limited to the verified 13 features `[VERIFIED BY SOURCE]`.
2. **Chronological Integrity:** Samples are stored as continuous temporal episodes to train the student GRU cell with realistic hidden-state dynamics `[PROPOSED]`.
3. **Split Isolation:** Train, Validation, and Test splits are divided strictly by **complete matches / episodes**, prohibiting adjacent-frame leakage `[PROPOSED]`.

---

## 2. Container Hierarchy and Data Organization

The dataset will be saved in versioned HDF5 format (`distillation_dataset_v1.h5`) or serialized PyTorch dictionary containers (`.pt`).

```text
distillation_dataset_v1.h5
├── /metadata
│   ├── dataset_id: "MW-DISTILL-NEXTO-20261011"
│   ├── schema_version: "1.0.0"
│   ├── collector_version: "1.0.0"
│   ├── teacher_sha256: "bf5343b5eeacac6bf7cdb75dac4a5c14ba0f94d820eae75f00a211b6119d69fa"
│   ├── student_architecture: "Linear(13->64)-ReLU-GRU(64->64)-Linear(64->5)"
│   └── total_episodes: int
└── /episodes
    ├── /ep_00001
    │   ├── /metadata (match_id, team, arena, goal_outcome)
    │   ├── student_features_13: float32 [T, 13]
    │   ├── teacher_raw_action: int32 [T]
    │   ├── teacher_logits_90: float32 [T, 90]
    │   ├── teacher_controls_8: float32 [T, 8]
    │   ├── mapped_student_mode: int32 [T]
    │   ├── mapped_student_steer: float32 [T]
    │   ├── mapping_valid: bool [T]
    │   ├── on_ground: bool [T]
    │   └── context_labels: string [T]
    └── /ep_00002 ...
```

---

## 3. Detailed Field Definitions

### 3.1 Metadata Header
| Field Name | Type | Description |
| :--- | :---: | :--- |
| `match_id` | `string` | UUID of the RLBot match instance. |
| `episode_id` | `int32` | Goal-to-goal or kickoff-to-goal continuous sequence index. |
| `team` | `int32` | `0` for Blue Team (defending $-Y$), `1` for Orange Team (defending $+Y$). |
| `arena` | `string` | Arena name (e.g., `"DFHStadium"`). |
| `tick_rate` | `int32` | Frequency of recording ticks ($15\text{ Hz}$, $60\text{ Hz}$, or $120\text{ Hz}$). |

### 3.2 Timestep Tensor Fields
| Field Name | Shape & Type | Constraints / Normalization | Description |
| :--- | :---: | :---: | :--- |
| `student_features_13` | `float32[13]` | Must match [`src/runtime.py`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/src/runtime.py) | Exact 13 features: $[\Delta x, \Delta y, \Delta z, d, v, \Delta x_p, \Delta y_p, \Delta z_p, d_p, m_b, m_p, \Delta t, \Delta t_p]$. |
| `student_features_valid`| `bool` | `True` | Flag verifying that all 13 features are finite and within normal bounds. |
| `teacher_raw_action` | `int32` | $\in [0, 89]$ | Nexto's discrete action selection from its 90-action lookup table. |
| `teacher_logits_90` | `float32[90]` | Finite float | Unnormalized action logits from Nexto forward pass (used for soft KL distillation). |
| `teacher_controls_8` | `float32[8]` | Normalized controls | $[ \text{thr}, \text{str}, \text{pitch}, \text{yaw}, \text{roll}, \text{jmp}, \text{bst}, \text{hnd} ]$. |
| `mapped_student_mode` | `int32` | $\in \{0, 1, 2, 3\}$ | Target mode: `0` (Neutral), `1` (Chase), `2` (Jump), `3` (Front Dodge). |
| `mapped_student_steer` | `float32` | $\in [-1.0, 1.0]$ | Target continuous steer value. |
| `mapping_valid` | `bool` | Boolean | `True` if teacher action is representable by student decoder; `False` if unrepresentable (e.g. reverse, flight). |
| `on_ground` | `bool` | Boolean | Car wheels touching ground surface. Steer loss applied only when `True`. |
| `context_label` | `string` | Categorical | `"KICKOFF"`, `"OFFENSE"`, `"DEFENSE"`, `"RECOVERY"`, `"UNREPRESENTABLE"`. |

---

## 4. Sequence Organization for GRU Training

The student policy architecture requires a 64-dimensional hidden state:
$$\mathbf{h}_t = \text{GRU}(\mathbf{h}_{t-1}, \mathbf{z}_t)$$

To properly condition the GRU:
1. **Truncated BPTT Chunks:** Timesteps are grouped into sequential windows of length $L = 64$ ticks ($\approx 4.27$ seconds at 15 Hz, or $1.07$ seconds at 60 Hz).
2. **Hidden State Resets:** At $t = 0$ of each new episode or after a goal reset, $\mathbf{h}_0$ is initialized strictly to zeros ($\mathbf{0} \in \mathbb{R}^{1 \times 1 \times 64}$).
3. **Carry-Over Dynamics:** Within an active episode, hidden states are propagated forward sequentially between adjacent windows.

---

## 5. Dataset Validation & Integrity Checks

Before training can be approved, the candidate dataset must pass the automated integrity suite in `scratch/distillation/validate_dataset.py`:
1. **NaN / Inf Check:** All tensors scanned for non-finite values. Any sample containing NaN is dropped.
2. **Feature Alignment Parity:** A random sample of 1,000 frames is recomputed using [`observation_contract.py:build()`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/src/observation_contract.py) to guarantee that recorded features match the official runtime to within machine precision ($|f_{\text{rec}} - f_{\text{live}}| < 10^{-6}$).
3. **Split Purity Verification:** Confirms that no `match_id` or `episode_id` appears in more than one split (Train: 70%, Validation: 15%, Test: 15%).
4. **Class Imbalance Distribution:** Mode frequencies are computed. Mode 1 (Chase) is expected to dominate (~75%); loss reweighting must be calculated based on empirical counts.
5. **Masking Verification:** Confirms that all frames with Nexto reverse throttle (`thr < 0`) or aerial flight (`jmp=1, bst=1, pitch=1`) have `mapping_valid == False`.

---

## 6. Stage A Sign-Off
- **Status:** Specification complete.
- **Stage B Prerequisite:** The offline collector harness must implement this schema exactly without modifying production participant files `[REQUIRED APPROVAL]`.
