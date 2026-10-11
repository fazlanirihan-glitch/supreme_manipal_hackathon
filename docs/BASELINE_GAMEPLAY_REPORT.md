# Model Wars: Baseline Gameplay Report
**Document ID:** `MW-REP-BASE-001`  
**Target:** Production Baseline (Candidate B — `candidate_B_steer_heavy.pt`)  
**SHA-256:** `bf434bfec1b5890af2847ec5c116e2f424bb8ce601a4894ed59091afc1ac645a`  
**Status:** Certified Stage A Baseline Audit  
**Classification Standard:** `[VERIFIED BY SOURCE]`, `[VERIFIED BY EXECUTION]`, `[EXPERIMENTAL RESULT]`, `[INFERRED]`, `[UNRESOLVED]`, `[REQUIRED APPROVAL]`

---

## 1. Executive Summary
This report establishes the verified baseline status of the Model Wars participant bot (`fazlanirihan-glitch/zencro-model-wars`) currently running Candidate B sealed in `model/V13.pt`.

An empirical audit of the repository reveals:
1. **Static Validation & Preflight:** The runtime, model architecture, tensor shapes, and submission credentials pass 100% of official tests (`[VERIFIED BY EXECUTION]`).
2. **Live Match Execution Status:** Real-time GUI match discovery, GUI launch, and live game telemetry have **NOT BEEN TESTED OR RECORDED** in the official `logs/` directory (`[VERIFIED BY SOURCE & EXECUTION]`). Previous benchmark claims in `PARTICIPANT_RESULTS.md` reflect offline synthetic kinematic simulations, not live Rocket League gameplay (`[VERIFIED BY SOURCE]`).
3. **Core Behavioral Failure Modes:** In synthetic kinematic testing, Candidate B successfully resolves turnaround deadlocks and suppresses errant aerial dodges. However, because the 13-dimensional observation vector lacks global pitch coordinates and team goal vectors, the model remains geometrically blind to net orientation, causing severe defensive vulnerability (`[VERIFIED BY SOURCE & INFERRED]`).

---

## 2. Runtime Launch and Infrastructure Verification

### 2.1 Bot Entrypoint and RLBot Interface
- **Entrypoint:** [`src/bot.py`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/src/bot.py) implements the native RLBot 2.0 interface subclassing `rlbot.managers.Bot` `[VERIFIED BY SOURCE]`.
- **Initialization:** Upon invocation, `initialize()` constructs an instance of `Runtime()` from [`src/runtime.py`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/src/runtime.py) `[VERIFIED BY SOURCE]`.
- **Checkpoint Loading:** `Runtime` loads `model/V13.pt` strictly onto CPU using `weights_only=True` and verifies that the checkpoint SHA-256 exactly matches the hardcoded `PIN` (`bf434bfe...`) `[VERIFIED BY SOURCE]`.
- **Output Validation:** The `submit` wrapper strictly verifies that all submitted controls originate from the student policy and match the internal expectation, raising a `RuntimeError` if any unauthorized controller mutation occurs `[VERIFIED BY SOURCE: bot.py L32-36]`.

### 2.2 Live Match Testing Status
- **Official Preflight Output:** Running `python src/preflight.py` explicitly reports:
  ```json
  {
    "status": "non_live_preflight_pass_not_GUI_acceptance",
    "GUI_discovery": "NOT TESTED",
    "GUI_launch": "NOT TESTED",
    "gameplay": "NOT TESTED",
    "no_live": true
  }
  ```
  `[VERIFIED BY EXECUTION]`.
- **Telemetry Event Logs:** [`src/bot.py`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/src/bot.py#L23-L27) writes runtime telemetry to `logs/{timestamp}_{pid}.jsonl`. Inspection of `Model-Wars-Participant/logs` confirms **zero log files exist**, confirming that the bot has never completed an official live GUI match within this workspace `[VERIFIED BY EXECUTION]`.

---

## 3. Observation & Action Decoder Operation

### 3.1 Observation Pipeline
- **Observation Space:** Strictly 13 continuous floating-point features constructed by `observation_contract.py:build()` `[VERIFIED BY SOURCE]`.
- **Ego-Centric Relativity:** Features $f_0 \dots f_3$ represent the relative ball vector in the car's local basis frame divided by 4000/2000 UU. Features $f_5 \dots f_8$ represent the 2.0-second predicted ball intercept. Feature $f_4$ represents car scalar speed divided by 2300 UU/s `[VERIFIED BY SOURCE]`.
- **Recurrent State:** Maintained across frames as a GRU hidden state $\mathbf{h} \in \mathbb{R}^{1 \times 1 \times 64}$. Zeroed on episode resets or long timing gaps `[VERIFIED BY SOURCE]`.

### 3.2 Action Decoder Contract
Model outputs $\mathbf{y} = [y_0, y_1, y_2, y_3, y_4]$:
- **Discrete Mode:** $\text{mode} = \arg\max(y_0, y_1, y_2, y_3) \in \{0, 1, 2, 3\}$.
- **Continuous Steering:** $\text{steer} = \text{clamp}(y_4, -1.0, 1.0)$.
- **Controller Decoded Channels:**
  - `Mode 0 (Neutral)`: `throttle = 0.0, steer = 0.0, jump = False, pitch = 0.0`
  - `Mode 1 (Chase)`: `throttle = 1.0, steer = steer, jump = False, pitch = 0.0`
  - `Mode 2 (Jump)`: `throttle = 1.0, steer = steer, jump = True, pitch = 0.0`
  - `Mode 3 (Front Dodge)`: `throttle = 1.0, steer = steer, jump = True, pitch = -1.0`
- **Restrictions:** `boost = False`, `handbrake = False`, `yaw = 0.0`, `roll = 0.0`, `throttle \ge 0.0` `[VERIFIED BY SOURCE: runtime.py L106-115]`.

---

## 4. Quantitative Benchmark Performance

Because live GUI gameplay has not been recorded, historical quantitative metrics are derived from the standardized kinematic benchmark suite in [`experiments/evaluator.py`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/experiments/evaluator.py).

### 4.1 Comparative Candidate Matrix (Synthetic Simulation)
| Metric | Baseline V13 (`579fa...`) | Candidate A (`68af4...`) | Candidate B (`bf434...`) [Current] | Candidate C (`3621e...`) |
| :--- | :---: | :---: | :---: | :---: |
| **Kickoff Final Speed (120 ticks)** | 1410.0 UU/s | 1410.0 UU/s | 1410.0 UU/s | 1410.0 UU/s |
| **Kickoff Distance Covered** | 1200.5 UU | 1262.3 UU | 1262.3 UU | 1262.3 UU |
| **Kickoff Deadlock (Mode 0 Ticks)** | 30 ticks | 0 ticks | 0 ticks | 0 ticks |
| **Turnaround Mode 0 Stalls** | 30 ticks | 0 ticks | 0 ticks | 0 ticks |
| **Turnaround Average Steer Lock** | 0.120 | 0.842 | **0.895** | 0.781 |
| **High Ball Aerial Dodges (Mode 3)** | 2 dodges | 0 dodges | 0 dodges | 0 dodges |
| **Validation Loss** | N/A | **0.2621** | 0.5098 | 0.2953 |
| **Mode Classification Accuracy** | N/A | 91.12% | 92.15% | **93.46%** |
| **Steering MAE** | N/A | **0.2300** | 0.2348 | 0.2803 |
| **Baseline Cosine Similarity** | 1.00000 | 0.93157 | 0.96424 | 0.91063 |

`[EXPERIMENTAL RESULT: experiments/evaluator.py & PARTICIPANT_RESULTS.md]`

---

## 5. Gameplay Failure Analysis

1. **Kickoff Acceleration:**
   - Candidate B eliminates the 30-tick hesitation present in Baseline V13. However, without boost (`boost = False`), the bot is fundamentally capped at 1410 UU/s, meaning any opponent utilizing boost will beat our student to the first touch `[VERIFIED BY SOURCE & INFERRED]`.
2. **Turnaround and Wall Recovery:**
   - In synthetic tests, Candidate B maintains steering lock and full throttle when the ball is behind. However, because the runtime cannot reverse (`throttle < 0`) or powerslide (`handbrake = True`), wide turning radiuses cause wall collisions and trapped situations in real corners `[INFERRED]`.
3. **Defensive Inability & Own Goals:**
   - **Root Cause:** Symmetrical ball positions relative to the car generate identical 13D observations regardless of whether the car is facing into its own net or the opponent's net. No amount of policy training on 13D observations can disambiguate net orientation without runtime coordinate overrides `[VERIFIED BY SOURCE]`.

---

## 6. Stage A Audit Conclusions & Recommendations
- **Baseline Integrity:** Candidate B is verified intact and correctly sealed in `model/V13.pt`.
- **Live GUI Requirement:** Before promoting any distilled candidate in later stages, live match testing against the built-in Psyonix bot must be executed to gather genuine game telemetry `[REQUIRED APPROVAL]`.
- **Preservation Notice:** Candidate B must remain untouched throughout all subsequent experimentation.
