# Model Wars Experiment & Provenance Tracker
**Document ID:** `MW-LOG-EXP-001`  
**Maintainer:** Model Wars Engineering Team  
**Scope:** Checkpoint Hashes, Metrics Registry & Provenance Tracking  
**Classification Standard:** `[VERIFIED BY SOURCE]`, `[VERIFIED BY EXECUTION]`, `[EXPERIMENTAL RESULT]`, `[INFERRED]`, `[UNRESOLVED]`, `[REQUIRED APPROVAL]`

---

## 1. Executive Summary
This document tracks all historical, active, and prospective checkpoints in the `Model-Wars-Participant` repository. It provides an unalterable audit trail ensuring that:
1. Every candidate has a recorded SHA-256 hash `[VERIFIED BY EXECUTION]`.
2. No candidate is overwritten during experimentation `[VERIFIED BY SOURCE]`.
3. Candidate B (`bf434bfe...`) remains the certified production model currently sealed in `model/V13.pt` `[VERIFIED BY EXECUTION]`.
4. Cosine similarity against the starter baseline remains below the competition plagiarism threshold of $0.999$ `[VERIFIED BY EXECUTION]`.

---

## 2. Checkpoint Registry

| Checkpoint Name | File Path | SHA-256 Hash | Size (Bytes) | Params | Cosine Sim | Val Loss | Mode Acc | Steer MAE | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Starter Baseline** | `experiments/baseline/V13_baseline.pt` | `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7` | 110,335 | 26,181 | 1.00000 | N/A | N/A | N/A | **Preserved Starter** `[VERIFIED]` |
| **Candidate A** | `experiments/checkpoints/candidate_A_balanced.pt` | `68af4e881b02d9686c3179ee5fd6b1831f41a7a06c53c50ea0133b732bcaac64` | 108,547 | 26,181 | 0.93157 | **0.2621** | 91.12% | **0.2300** | **Preserved Checkpoint** `[VERIFIED]` |
| **Candidate B (Production)** | `model/V13.pt` & `experiments/checkpoints/candidate_B_steer_heavy.pt` | `bf434bfec1b5890af2847ec5c116e2f424bb8ce601a4894ed59091afc1ac645a` | 108,653 | 26,181 | 0.96424 | 0.5098 | 92.15% | 0.2348 | **ACTIVE SEALED PRODUCTION** `[VERIFIED]` |
| **Candidate C** | `experiments/checkpoints/candidate_C_rapid_adapt.pt` | `3621edfe1bcd43b8abb6d47c1d386a42abc9c05130f0cbb1859069ea316f09cc` | 108,653 | 26,181 | 0.91063 | 0.2953 | **93.46%** | 0.2803 | **Preserved Checkpoint** `[VERIFIED]` |
| **Candidate D (Distilled)** | `experiments/checkpoints/candidate_D_distilled.pt` | *Pending Stage D Training* | *Pending* | 26,181 | *TBD* | *TBD* | *TBD* | *TBD* | **PROPOSED** `[REQUIRED APPROVAL]` |

---

## 3. Training Run Configurations (Preserved Candidates)

### Run 1: Candidate A (Balanced Multi-Task)
- **Dataset:** `GoalAwareExpert_Simulation_V1` (50 episodes, 12,000 frames)
- **Objective:** Weighted Cross-Entropy + $2.0 \times \text{SmoothL1}$ Steer
- **Optimizer:** AdamW ($lr = 3 \times 10^{-4}$, weight decay $10^{-4}$)
- **Epochs:** 15 epochs
- **Strengths:** Lowest overall validation loss (0.2621) and lowest steering MAE (0.2300).
- **Limitation:** Slightly looser mode certainty under pressure than Candidate B.

### Run 2: Candidate B (Steer-Heavy Regularization — Active Production)
- **Dataset:** `GoalAwareExpert_Simulation_V1` (50 episodes, 12,000 frames)
- **Objective:** Cross-Entropy + $4.0 \times \text{SmoothL1}$ Steer
- **Optimizer:** AdamW ($lr = 3 \times 10^{-4}$, weight decay $10^{-4}$)
- **Epochs:** 20 epochs
- **Strengths:** Extreme steering stability (steer weight 4.0); highest turnaround lock (0.895); zero turnaround stalls; compliant cosine similarity 0.96424.
- **Current Role:** Sealed in `model/V13.pt`; hardcoded PIN in `src/runtime.py` L38.

### Run 3: Candidate C (Rapid Adaptation)
- **Dataset:** `GoalAwareExpert_Simulation_V1` (50 episodes, 12,000 frames)
- **Objective:** Weighted Cross-Entropy + $2.5 \times \text{SmoothL1}$ Steer
- **Optimizer:** AdamW ($lr = 5 \times 10^{-4}$, weight decay $10^{-4}$)
- **Epochs:** 25 epochs
- **Strengths:** Highest discrete mode accuracy (93.46%).
- **Limitation:** Higher steering MAE (0.2803) caused minor steering jitter.

---

## 4. Hardware, Runtime & Dependency Environment

| Component | Verified Specification |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 (64-bit) `[VERIFIED BY EXECUTION]` |
| **Python Runtime** | Python 3.12.7 `[VERIFIED BY EXECUTION]` |
| **PyTorch Version** | `torch 2.14.1+cpu` `[VERIFIED BY EXECUTION]` |
| **NumPy Version** | `numpy 1.26.4` `[VERIFIED BY EXECUTION]` |
| **RLBot Version** | `rlbot 2.0.0-beta.55` `[VERIFIED BY EXECUTION]` |
| **FlatBuffers Version** | `rlbot-flatbuffers 0.19.0` `[VERIFIED BY EXECUTION]` |
| **Compute Device** | Intel CPU (Zero CUDA dependency) `[VERIFIED BY SOURCE]` |

---

## 5. Provenance Files Registry

1. [`candidate_manifest.json`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/candidate_manifest.json): Records hashes of all 6 runtime files and `checkpoint_sha256 = bf434bfe...`.
2. [`submission.json`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/submission.json): Records `agent_id = fazlanirihan-glitch/zencro-model-wars`, `checkpoint_sha256 = bf434bfe...`, and training provenance dictionary.
3. [`docs/PARTICIPANT_RESULTS.md`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/docs/PARTICIPANT_RESULTS.md): Preserves quantitative experiment logs for Candidates A, B, and C.
4. [`experiments/provenance_candidate_b.json`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/experiments/provenance_candidate_b.json): Standalone training metadata and random seed record for Candidate B.
