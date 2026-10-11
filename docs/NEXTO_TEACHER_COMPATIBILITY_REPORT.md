# Nexto Teacher Compatibility and Architecture Report
**Document ID:** `MW-REP-NEXTO-001`  
**Reference Teacher:** Nexto Pretrained Bot (`NectoFamily-master`)  
**Target:** Policy Distillation Feasibility Audit  
**Classification Standard:** `[VERIFIED BY SOURCE]`, `[VERIFIED BY EXECUTION]`, `[EXPERIMENTAL RESULT]`, `[INFERRED]`, `[UNRESOLVED]`, `[REQUIRED APPROVAL]`

---

## 1. Executive Summary
This report documents the architectural, computational, and behavioral audit of the pretrained **Nexto** model located in `c:\Users\Saheer Hawa\OneDrive\Desktop\NectoFamily-master\NectoFamily-master\nexto`. 

Nexto is an expert-tier Rocket League bot developed by Rolv-Arild and the RLGym community. While Nexto exhibits superhuman ball-control and mechanics in full-state environments, our audit confirms that:
1. **Model Parameter Asymmetry:** Nexto has **444,032 parameters** ($\approx 17\times$ the Student's strict budget of **26,181 parameters**) `[VERIFIED BY EXECUTION]`.
2. **Observation Space Mismatch:** Nexto receives **37 entities $\times$ 24 features** (full 3D global field coordinates, opponent tracking, and 34 boost pads with cooldown timers). The Student receives only **13 scalar features** (car-relative ball displacement only). Over 85% of Nexto's sensory inputs are absent from the student contract `[VERIFIED BY SOURCE]`.
3. **Action Space Mismatch:** Over 70% of Nexto's 90 discrete actions rely on controls permanently forbidden or hardcoded in the student decoder (`boost`, `handbrake`, `reverse`, `roll`, `aerial pitch-up`) `[VERIFIED BY SOURCE]`.
4. **Environment Isolation Requirement:** Nexto requires `numpy==2.*` and `rlgym_compat`, which conflict with the Student's runtime (`numpy==1.26.4`). Any data collection must occur in an isolated harness (`scratch/distillation/`) `[VERIFIED BY SOURCE]`.

---

## 2. Checkpoint Provenance and Integrity

| Property | Value | Audit Verification |
| :--- | :--- | :---: |
| **File Location** | `NectoFamily-master/nexto/nexto-model.pt` | `[VERIFIED BY SOURCE]` |
| **File Size** | `1,852,625` bytes (~1.77 MB) | `[VERIFIED BY EXECUTION]` |
| **SHA-256 Hash** | `bf5343b5eeacac6bf7cdb75dac4a5c14ba0f94d820eae75f00a211b6119d69fa` | `[VERIFIED BY EXECUTION]` |
| **Container Format** | PyTorch TorchScript `RecursiveScriptModule` | `[VERIFIED BY EXECUTION]` |
| **Trainable Parameters**| **444,032** across multi-head cross-attention & MLP blocks | `[VERIFIED BY EXECUTION]` |
| **Upstream Repository** | `github.com/Rolv-Arild/Necto` | `[VERIFIED BY SOURCE]` |
| **Upstream Licence** | No LICENSE file in repo ("All Rights Reserved") | `[VERIFIED BY SOURCE]` |
| **Organizer Clearance**| Permitted provided final submission complies with Model Wars rules | `[REQUIRED APPROVAL]` |

---

## 3. Teacher Architecture and Forward Pass Specification

Nexto uses a transformer architecture designed for multi-entity sports simulations.

### 3.1 Input Tensor Structure
The model expects a 3-element tuple `(q, kv, mask)`:
1. **Query Tensor ($\mathbf{q}$):** Shape `[batch, 1, 32]`.
   - Represents the ego car's internal state.
   - Contains 3D velocity, angular velocity, boost fraction, ground contact flag, flip availability flag, and the 8 previous continuous controller inputs `[VERIFIED BY SOURCE: nexto/nexto_obs.py L120-155]`.
2. **Key-Value Tensor ($\mathbf{kv}$):** Shape `[batch, N_entities, 24]`.
   - In 1v1 gameplay, $N_{\text{entities}} = 37$:
     - Token 0: Ego car (relative to world).
     - Token 1: Opponent car (relative position, velocity, orientation, boost state).
     - Token 2: Ball (relative position, velocity, angular velocity).
     - Tokens 3–36: **34 boost pads** with 3D positions, pad type (small/large), active boolean flag, and respawn timer `[VERIFIED BY SOURCE: nexto/nexto_obs.py L180-240]`.
3. **Attention Mask Tensor ($\mathbf{mask}$):** Shape `[batch, 1, 37]`.
   - Boolean / float tensor indicating valid entities.

### 3.2 Output Tensor Structure
Nexto returns a 2-element tuple `(out, weights)`:
- `out`: Float tensor of shape `[batch, 90]` representing unnormalized log-probabilities (logits) across 90 discrete actions.
- `weights`: Attention weight tensor of shape `[batch, heads, 1, 37]`.

---

## 4. Decision Making and Action Lookup Table

### 4.1 Inference Timing and Tick Rate
- **Game Engine Rate:** Rocket League physics ticks at 120 Hz.
- **Tick Skip:** `nexto/bot.py` specifies `tick_skip = 8` `[VERIFIED BY SOURCE: nexto/bot.py L42]`.
- **Decision Frequency:**
  $$f_{\text{decision}} = \frac{120\text{ Hz}}{8} = 15\text{ Hz} \quad (\Delta t = 66.67\text{ ms})$$
- Note: The Student runtime processes every callback ($60\text{ Hz} \text{ or } 120\text{ Hz}$, $\Delta t \approx 8.33\text{--}16.67\text{ ms}$). A teacher demonstration recorder must hold or interpolate Nexto decisions across student frames `[VERIFIED BY SOURCE & INFERRED]`.

### 4.2 Action Selection Logic
In `nexto/agent.py`:
- **Competitive Mode:** `action_index = np.argmax(out)` (deterministic greedy selection).
- **Stochastic Mode:** Categorical sampling with temperature:
  $$P(a_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- **Action Decoder (`make_lookup_table()`):** Maps integer index $i \in [0, 89]$ to an 8-dimensional controller array:
  $$[\text{throttle}, \text{steer}, \text{pitch}, \text{yaw}, \text{roll}, \text{jump}, \text{boost}, \text{handbrake}]$$
  `[VERIFIED BY SOURCE: nexto/agent.py L90-128]`.

### 4.3 Kickoff State Machine
Nexto contains an explicit hardcoded kickoff module (`Speedkickoff` in `nexto/bot.py`) that executes a scripted speed-flip kickoff when `packet.game_info.is_kickoff_pause` is active. This behavior does **not** originate from the neural network weights `[VERIFIED BY SOURCE]`.

---

## 5. Compatibility Bottleneck Summary

| Dimension | Nexto Teacher | Model Wars Student | Compatibility Assessment |
| :--- | :--- | :--- | :--- |
| **Parameters** | 444,032 | 26,181 | **16.96$\times$ larger**; cannot be transferred directly. |
| **Architecture** | Entity Attention Transformer | 1-Layer Linear $\to$ GRU $\to$ Linear | Fundamentally different graph topology. |
| **Observation Channels**| $37 \times 24 + 32 = 920$ inputs | Exactly 13 scalar inputs | **Severe Information Bottleneck** (98.6% channel reduction). |
| **Global Field Bounds** | Aware via boost pad positions | None | Student cannot sense field boundary proximity. |
| **Goal Locations** | Explicit global $Y = \pm 5120$ | None | Student cannot sense net orientation from inputs. |
| **Opponent Tracking** | Full 3D tracking | None | Student is blind to opponent positioning. |
| **Actuation Range** | 90 discrete actions (full 8 channels) | 4 discrete modes + continuous steer | Over 70% of Nexto actions unrepresentable. |

---

## 6. Stage A Conclusions & Prerequisites for Stage B
1. **Zero Runtime Intrusion:** Nexto code, weights, or dependencies must **never** be imported into `Model-Wars-Participant/src/` `[VERIFIED BY SOURCE]`.
2. **Filtered Distillation Only:** Because over 70% of Nexto actions are unrepresentable by the student decoder, teacher demonstrations must be strictly filtered to ground steering and defensive approach angles `[PROPOSED]`.
3. **Isolated Collection Harness:** All teacher data collection must be contained in `scratch/distillation/` or `experiments/distillation/` using a separate environment with `rlgym_compat` `[REQUIRED APPROVAL]`.
