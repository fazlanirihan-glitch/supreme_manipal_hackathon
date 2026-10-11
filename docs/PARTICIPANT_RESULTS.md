# ZenCro Model Wars — Participant Training & Validation Results

## 1. Executive Summary & Architecture
- **Participant Team**: ZenCro Model Wars
- **Agent ID**: `fazlanirihan-glitch/zencro-model-wars`
- **Network Architecture**: 13D GRU Policy (`Linear(13, 64) -> ReLU -> GRU(64, 64) -> Linear(64, 5)`)
- **Parameters**: 26,181 trainable parameters (100% compliant with official contract)
- **Primary Checkpoint**: Candidate B (`candidate_B_steer_heavy.pt`)

## 2. Experimental Data Pipeline
- **Dataset**: `GoalAwareExpert_Simulation_V1`
- **Sample Count**: 50 contiguous match episodes (12,000 frames total at 60 Hz)
- **Scenarios Covered**:
  - Symmetric Kickoffs (center, diagonal, corner alignments)
  - Offensive Interception & Attacks toward opponent net ($Y = \pm 5120$)
  - Defensive Arcing Clearances (swinging around the ball to prevent own-goal deflections)
  - Behind-the-car Turnarounds (re-engaging ball with full forward throttle and steering lock)
  - Aerial Ball Ground Positioning (suppressing counterproductive ground front dodges)

## 3. Quantitative Experiment Results
Across 3 controlled candidate runs:

| Candidate | Objective / Loss | Val Loss | Mode Acc | Steer MAE | Cosine Similarity | Turnaround Stalls | Aerial Ground Dodges |
|---|---|---|---|---|---|---|---|
| **Baseline V13** | N/A (Starter) | N/A | N/A | N/A | 1.00000 | 30 ticks | 2 dodges |
| **Candidate A** | Weighted CE + 2.0*SmoothL1 | 0.2621 | 91.12% | 0.2300 | 0.93157 | 0 ticks | 0 dodges |
| **Candidate B (Selected)** | CE + 4.0*SmoothL1 | 0.5098 | 92.15% | 0.2348 | 0.96424 | 0 ticks | 0 dodges |
| **Candidate C** | Weighted CE (lr=5e-4) + 2.5*SmoothL1 | 0.2953 | 93.46% | 0.2803 | 0.91063 | 0 ticks | 0 dodges |

## 4. Key Gameplay Improvements
1. **Turnaround Stall Elimination**: Zero Mode 0 deadlocks when the ball passes behind the car (baseline stalled for 30 ticks).
2. **Aerial Dodge Suppression**: Zero erroneous ground front flips under high balls (baseline committed 2 ground flips under 1500 UU balls).
3. **Smooth Continuous Steering**: Steering MAE reduced to 0.2348 with high-weight regularization.
4. **Legitimate Weight Differentiation**: Baseline cosine similarity is 0.96424, compliant with the official $\le 0.999$ threshold.
