# Model Wars Fair Gameplay Evaluation Protocol
**Document ID:** `MW-PROT-EVAL-001`  
**Purpose:** Standardized Match Protocol & Candidate Promotion Criteria  
**Benchmark Target:** Unchanged Candidate B (`candidate_B_steer_heavy.pt`)  
**Classification Standard:** `[VERIFIED BY SOURCE]`, `[VERIFIED BY EXECUTION]`, `[EXPERIMENTAL RESULT]`, `[INFERRED]`, `[UNRESOLVED]`, `[REQUIRED APPROVAL]`

---

## 1. Evaluation Philosophy and Core Mandate
Under the Master Engineering Directive:
- **Offline metrics are diagnostic only:** Neither training loss, validation accuracy, nor synthetic kinematic simulation scores constitute proof of gameplay superiority `[VERIFIED BY SOURCE: IMPROVING_THE_BOT.md]`.
- **Actual gameplay is decisive:** A candidate model may only be promoted to production if it demonstrates statistically verified improvements in simulated or live Rocket League matches under standardized conditions `[PROPOSED]`.
- **Zero Regression Principle:** A candidate cannot be promoted if an improvement in offensive scoring is accompanied by an increase in own goals, wall stalling, or runtime exceptions `[PROPOSED]`.

---

## 2. Standardized Match Environment Settings

All comparative evaluation matches must be conducted under identical parameters:

| Parameter | Value | Compliance Reason |
| :--- | :--- | :--- |
| **Game Mode** | 1v1 Soccar | Official tournament format. |
| **Arena** | `DFHStadium` (Standard) | Standard geometry; prevents non-standard bounce bias. |
| **Match Duration** | 5 minutes (300 seconds regulation) | Full tournament length. |
| **Overtime** | Sudden death (+5 min max) | Resolves draws fairly. |
| **Mutator Settings** | Default / None | Normal gravity, normal ball weight, normal bounciness. |
| **Team Rotation** | 50% Blue / 50% Orange | Eliminates spawn side and diagonal kickoff asymmetry. |
| **Sample Size** | **10 matches minimum** per candidate pair | 5 matches Blue + 5 matches Orange to achieve statistical significance. |
| **Inference Hardware**| CPU only (`torch.device("cpu")`) | Matches official submission runtime environment. |

---

## 3. Candidate Comparison Matrix

Every test campaign evaluates the prospective candidate against our preserved candidate suite:
1. **Candidate B (Production Baseline):** `bf434bfec1b5890af2847ec5c116e2f424bb8ce601a4894ed59091afc1ac645a`
2. **Candidate A (Balanced Baseline):** `68af4e881b02d9686c3179ee5fd6b1831f41a7a06c53c50ea0133b732bcaac64`
3. **Candidate C (Rapid Adaptation Baseline):** `3621edfe1bcd43b8abb6d47c1d386a42abc9c05130f0cbb1859069ea316f09cc`
4. **Starter Baseline (V13 Original):** `579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`

---

## 4. Operational Definitions of Quantitative Metrics

To prevent subjective or unmeasured claims, every metric is computed programmatically from match telemetry:

### 4.1 Match Outcome Metrics
- **Win Rate ($\text{WR}$):** Fraction of completed matches won:
  $$\text{WR} = \frac{W + 0.5 \cdot D}{N_{\text{matches}}}$$
- **Goal Differential ($\Delta G$):**
  $$\Delta G = \frac{\sum G_{\text{scored}} - \sum G_{\text{conceded}}}{N_{\text{matches}}}$$

### 4.2 Tactical & Defensive Metrics
- **Own Goal Frequency ($\text{OG}$):**
  - *Definition:* Any touch by the bot where the ball crosses its own team's goal line within 3.0 seconds, with no subsequent opponent touch.
  - *Target:* $\text{OG} = 0$ per 5-minute match.
- **Useful Contact Rate ($\text{UCR}$):**
  - *Definition:* Percentage of ball contacts where the post-contact ball velocity vector increases toward the opponent's net ($\Delta v_y > 0$ for Blue, $\Delta v_y < 0$ for Orange) while the car is positioned goal-side of the ball.
  - *Target:* $> 65\%$ of touches.
- **Wall-Trapping Incident ($\text{WTI}$):**
  - *Definition:* Sustained period of $\ge 3.0$ consecutive seconds where car forward speed is $< 300$ UU/s and car distance to the side or back wall is $< 200$ UU.
  - *Target:* $< 1.0$ incident per match.
- **Turnaround Oscillation ($\text{TOO}$):**
  - *Definition:* An occurrence where the steering sign flips ($\text{sign}(s_t) \neq \text{sign}(s_{t-1})$) $\ge 4$ times within 1.5 seconds while car speed is $< 500$ UU/s and ball is behind the car.
  - *Target:* Zero oscillations.
- **Kickoff First-Touch Rate ($\text{KFTR}$):**
  - *Definition:* Percentage of kickoffs where the candidate contacts the ball before or simultaneously with the opponent.

### 4.3 Runtime Performance & Stability
- **Inference Latency:** Average forward pass time across all frames (must remain $< 1.5$ ms on CPU).
- **Failure Count:** Uncaught Python exceptions, NaN/Inf tensor outputs, or controller timeouts (must be **strictly 0**).

---

## 5. Formal Promotion Criteria

A new experimental candidate (e.g. Candidate D) may **only** replace Candidate B as the sealed model in `model/V13.pt` if all the following gates pass:

1. **Static Validation Gate:**
   - 100% pass on [`tools/validate_submission.py`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/tools/validate_submission.py).
   - 100% pass on [`src/preflight.py`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/src/preflight.py).
   - Parameter count is exactly **26,181**.
   - Cosine similarity to starter baseline is $\le 0.999$.
2. **Gameplay Superiority Gate:**
   - Candidate D achieves a **higher Win Rate** and **statistically superior Goal Differential** ($\Delta G$) against Candidate B across 10 head-to-head matches.
3. **Safety & Stability Gate:**
   - Own Goal rate does not exceed Candidate B.
   - Wall-trapping rate does not exceed Candidate B.
   - Zero runtime crashes across all test matches.

---

## 6. Stage A Sign-Off
- **Status:** Evaluation protocol and operational metric definitions formalized.
- **Stage E Prerequisite:** The above 10-match evaluation protocol must be executed before sealing any new weights `[REQUIRED APPROVAL]`.
