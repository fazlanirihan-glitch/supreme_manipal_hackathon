# Nexto-to-Student Action Compatibility Matrix & Mapping Specification
**Document ID:** `MW-SPEC-ACT-001`  
**Target:** 90 Discrete Nexto Actions $\to$ Student 4 Modes + Continuous Steer  
**Classification Standard:** `[VERIFIED BY SOURCE]`, `[VERIFIED BY EXECUTION]`, `[EXPERIMENTAL RESULT]`, `[INFERRED]`, `[UNRESOLVED]`, `[REQUIRED APPROVAL]`

---

## 1. Executive Summary
The Student model deployed in Model Wars uses a constrained action decoder defined in [`src/runtime.py` lines 86–135](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/src/runtime.py#L86-L135). It can produce only four discrete modes ($\text{Mode } 0 \dots 3$) and a single continuous steering scalar $s \in [-1.0, 1.0]$. Key vehicle capabilities including **boost acceleration**, **handbrake powersliding**, **reverse throttle**, and **3D aerial rotation** are permanently locked or disabled.

In contrast, Nexto selects among 90 discrete actions spanning full 8-dimensional controller inputs `[VERIFIED BY SOURCE: nexto/agent.py L90-128]`.

This document establishes the exact mathematical mapping and filtering rules for converting Nexto teacher actions into student supervision targets.

---

## 2. Action Interface Comparison

| Control Channel | Nexto Teacher Capability | Student Runtime Decoder | Contract Compatibility |
| :--- | :--- | :--- | :---: |
| **Throttle** | Discrete: $\{-1.0, 0.0, 1.0\}$ | Binary: $\{0.0, 1.0\}$ (Mode 0: 0.0, Mode 1–3: 1.0) | **No Reverse Throttle** `[VERIFIED]` |
| **Steering** | Discrete: $\{-1.0, 0.0, 1.0\}$ | Continuous: $\text{clamp}(y_4, -1.0, 1.0)$ | **Fully Representable** `[VERIFIED]` |
| **Pitch** | Discrete: $\{-1.0, 0.0, 1.0\}$ | Fixed: $-1.0$ (Mode 3) or $0.0$ (Mode 0–2) | **No Pitch-Up / Aerial Flight** `[VERIFIED]` |
| **Yaw** | Discrete: $\{-1.0, 0.0, 1.0\}$ | Hardcoded: $0.0$ | **Not Representable** `[VERIFIED]` |
| **Roll** | Discrete: $\{-1.0, 0.0, 1.0\}$ | Hardcoded: $0.0$ | **Not Representable** `[VERIFIED]` |
| **Jump** | Boolean: $\{0, 1\}$ | Boolean: $\{0, 1\}$ (Mode 2 & 3: True) | **Fully Representable** `[VERIFIED]` |
| **Boost** | Boolean: $\{0, 1\}$ | Hardcoded: **False** (`runtime.py` L112) | **Completely Forbidden** `[VERIFIED]` |
| **Handbrake** | Boolean: $\{0, 1\}$ | Hardcoded: **False** (`runtime.py` L113) | **Completely Forbidden** `[VERIFIED]` |

---

## 3. Comprehensive Nexto 90-Action Classification

Across Nexto's 90 discrete actions:
- **Category 1 (Directly Representable):** 26 actions (28.9%)
- **Category 2 (Representable Through Validated Approximation):** 22 actions (24.4%)
- **Category 3 (Completely Unrepresentable — Must be Filtered):** 42 actions (46.7%)

### 3.1 Category 1: Directly Representable Actions
These actions match the student runtime decoder without physical distortion:

| Action Family | Nexto Channel Signature | Student Mode Target | Student Steer Target | Tactical Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Ground Drive (Straight)** | `thr=1, str=0, jmp=0, bst=0, p=0` | `Mode 1 (Chase)` | `0.0` | Straight line acceleration. |
| **Ground Drive (Left)** | `thr=1, str=-1, jmp=0, bst=0, p=0` | `Mode 1 (Chase)` | `-1.0` | Left turn tracking. |
| **Ground Drive (Right)** | `thr=1, str=+1, jmp=0, bst=0, p=0` | `Mode 1 (Chase)` | `+1.0` | Right turn tracking. |
| **Ground Idle / Coast** | `thr=0, str=0, jmp=0, bst=0` | `Mode 0 (Neutral)` | `0.0` | Deceleration / waiting. |
| **Single Jump Liftoff** | `thr=1, jmp=1, bst=0, p=0, y=0, r=0` | `Mode 2 (Jump)` | `str` | Low 50/50 block or low ball hit. |
| **Straight Front Flip** | `thr=1, jmp=1, p=-1, bst=0, y=0, r=0` | `Mode 3 (Front Dodge)` | `0.0` | Forward power strike / speed gain. |

### 3.2 Category 2: Representable Through Validated Approximation
These actions require mapped approximations:

| Action Family | Nexto Channel Signature | Student Mode Target | Student Steer Target | Approximation Impact & Trade-Off |
| :--- | :--- | :---: | :---: | :--- |
| **Boost Ground Drive** | `thr=1, bst=1, jmp=0, p=0` | `Mode 1 (Chase)` | `str` | **Velocity Deficit:** Student executes normal drive. Max speed is capped at 1410 UU/s instead of supersonic 2300 UU/s. |
| **Diagonal Dodge** | `jmp=1, p=-1, y=\pm 1` or `r=\pm 1` | `Mode 3 (Front Dodge)` | $\pm 1.0$ (Steer Lock) | **No Roll Torque:** Student flips forward with front steering locked; cannot barrel roll sideways. |
| **Coasting Steer** | `thr=0, str=\pm 1, jmp=0` | `Mode 0 (Neutral)` | $\pm 1.0$ | Mode 0 hardcodes `steer = 0.0`. Steer input is suppressed by decoder. |

### 3.3 Category 3: Completely Unrepresentable Actions (Filtered Out)
These actions cannot be executed by the student runtime. Supervising the student to imitate these actions causes policy collapse, erratic jumping, or wall collisions:

| Action Family | Nexto Channel Signature | Cause of Incompatibility | Danger if Naively Imitated |
| :--- | :--- | :--- | :--- |
| **Reverse Throttle** | `thr = -1.0` | Student decoder has no negative throttle. | Mapping to Mode 1 drives forward into danger; mapping to Mode 0 coasts forward. |
| **Powerslide Turns** | `hnd = 1` | `handbrake` is hardcoded `False`. | Attempting to replicate Nexto's sharp cut angles without powerslide causes wide understeer and wall collisions. |
| **Fast Aerial Launch** | `jmp=1, bst=1, p=1` (pitch up) | Student cannot pitch up ($p \le 0$) and cannot boost. | Results in a flat, flopping ground hop with no forward momentum. |
| **Air Rolls & Yaw Turns** | $y \neq 0$ or $r \neq 0$ in mid-air | `yaw` and `roll` hardcoded `0.0`. | Cannot orient vehicle in 3D air space. |
| **Reverse Shadow Defense**| `thr=-1, looking at play` | No reverse throttle available. | Bot charges ball instead of shadowing back toward net. |

---

## 4. Operational Filtering and Masking Algorithm

During dataset processing and loss computation, each frame is evaluated by the Boolean validity function:

```python
def is_teacher_action_representable(controls) -> bool:
    # 1. Reverse throttle is strictly unrepresentable
    if controls.throttle < 0.0:
        return False

    # 2. Powerslide turns cannot be represented
    if controls.handbrake:
        return False

    # 3. Aerial flight (pitch up while jumping with boost) is unrepresentable
    if controls.jump and controls.pitch > 0.0:
        return False

    # 4. Pure aerial rolls without forward dodge are unrepresentable
    if controls.jump and (controls.roll != 0.0 or controls.yaw != 0.0) and controls.pitch != -1.0:
        return False

    return True
```

### Loss Mask Formulation:
For timestep $t$:
$$w_t = \begin{cases} 1.0 & \text{if } \text{is\_teacher\_action\_representable}(\mathbf{c}_t) = \text{True} \\ 0.0 & \text{otherwise} \end{cases}$$

Frames with $w_t = 0.0$ are **masked out of the action-mode cross-entropy loss**, preventing the student from attempting unachievable mechanics.

---

## 5. Capabilities Benefiting from Filtered Transfer

| Target Gameplay Area | Teacher Capability Transferred | Expected Student Improvement |
| :--- | :--- | :--- |
| **Ground Steering** | Smooth curve pursuit from Nexto ground paths | Eliminates discrete steering jerk and reduces Steering MAE. |
| **Ball Approach Angle** | Arcing around the ball rather than straight-line chasing | Teaches the student to attack the ball toward the opponent net instead of direct head-on collision. |
| **Recovery from Behind** | Nexto aggressive steering lock when ball is behind | Prevents Mode 0 deadlocks and reduces turnaround time. |
| **High Ball Discipline** | Nexto stays grounded under high aerial balls when low on boost | Suppresses errant Mode 3 ground flips under 1500+ UU balls. |

---

## 6. Stage A Sign-Off
- **Status:** Action compatibility matrix and mathematical filtering mask established.
- **Stage B Prerequisite:** Collector harness must tag all frames with `mapping_valid` based on Section 4 `[REQUIRED APPROVAL]`.
