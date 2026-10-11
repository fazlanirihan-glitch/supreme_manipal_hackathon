# Nexto Policy Distillation Training Plan
**Document ID:** `MW-PLAN-TRAIN-001`  
**Status:** Design Only — Pending Stage D Execution Approval  
**Target Model:** Candidate D (`experiments/checkpoints/candidate_D_distilled.pt`)  
**Classification Standard:** `[VERIFIED BY SOURCE]`, `[VERIFIED BY EXECUTION]`, `[EXPERIMENTAL RESULT]`, `[INFERRED]`, `[UNRESOLVED]`, `[REQUIRED APPROVAL]`

---

## 1. Architectural Invariants and Constraints

Under the Model Wars competition contract, the student network architecture must remain strictly:
$$\mathbf{z} = \text{ReLU}(W_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}), \quad W_{\text{enc}} \in \mathbb{R}^{64 \times 13}$$
$$\mathbf{h}_t = \text{GRU}(\mathbf{z}, \mathbf{h}_{t-1}), \quad \text{hidden size } 64$$
$$\mathbf{y}_t = W_{\text{head}} \mathbf{h}_t + \mathbf{b}_{\text{head}}, \quad W_{\text{head}} \in \mathbb{R}^{5 \times 64}$$

**Non-Negotiable Constraints:**
- **Input Dimensions:** Exactly 13 scalar features `[VERIFIED BY SOURCE: policy.py]`.
- **Trainable Parameters:** Exactly **26,181** `[VERIFIED BY EXECUTION]`.
- **Output Dimensions:** Exactly 5 scalar logits `[VERIFIED BY SOURCE: policy.py]`.
- **No Extra Heads or Branches:** Auxiliary loss heads cannot be kept at inference time.

---

## 2. Loss Formulation and Multi-Task Objective

The model outputs 5 continuous values $\mathbf{y}_t = [y_{t,0}, y_{t,1}, y_{t,2}, y_{t,3}, y_{t,4}]$:
- Indices $0 \dots 3$: Unnormalized mode logits for Neutral, Chase, Jump, Front Dodge.
- Index $4$: Continuous steer target $\in [-1.0, 1.0]$.

The combined training loss is:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{mode}} + \beta \mathcal{L}_{\text{steer}} + \gamma \mathcal{L}_{\text{temp}}$$

### 2.1 Masked Class-Weighted Cross-Entropy ($\mathcal{L}_{\text{mode}}$)
To account for class imbalance (where Chase mode accounts for $>70\%$ of frames) and to filter unrepresentable teacher actions:
$$\mathcal{L}_{\text{mode}} = - \frac{1}{\sum_{t=1}^T w_t} \sum_{t=1}^T w_t \cdot C(m_t^*) \cdot \log \left( \frac{\exp(y_{t, m_t^*})}{\sum_{j=0}^3 \exp(y_{t, j})} \right)$$
Where:
- $m_t^* \in \{0, 1, 2, 3\}$ is the mapped teacher mode.
- $w_t \in \{0.0, 1.0\}$ is the validity mask from [`ACTION_COMPATIBILITY_MATRIX.md`](file:///c:/Users/Saheer%20Hawa/OneDrive/Desktop/Model-Wars-Base-Repository-main/Model-Wars-Participant/docs/ACTION_COMPATIBILITY_MATRIX.md) ($w_t = 0.0$ if teacher executed reverse throttle, powerslide, or aerial boost).
- $C(m)$ is the inverse frequency class weight vector:
  $$C = [C_{\text{Neutral}} = 2.0, \quad C_{\text{Chase}} = 1.0, \quad C_{\text{Jump}} = 3.0, \quad C_{\text{Dodge}} = 2.5]$$

### 2.2 Ground-Contact Steering Regression ($\mathcal{L}_{\text{steer}}$)
Steering error is supervised using Smooth L1 (Huber) loss, but **only when wheels are in contact with the ground** ($g_t = 1$):
$$\mathcal{L}_{\text{steer}} = \frac{1}{\sum_{t=1}^T g_t} \sum_{t=1}^T g_t \cdot \text{SmoothL1}(y_{t, 4}, s_t^*)$$
Where:
$$\text{SmoothL1}(d) = \begin{cases} 0.5 d^2 & \text{if } |d| < 1.0 \\ |d| - 0.5 & \text{otherwise} \end{cases}$$
Airborne frames ($g_t = 0$) do not penalize steering because car wheels have zero ground traction in mid-air.

### 2.3 Temporal Consistency Regularization ($\mathcal{L}_{\text{temp}}$)
Prevents high-frequency mode chatter (e.g. oscillating between Chase and Jump):
$$\mathcal{L}_{\text{temp}} = \frac{1}{T-1} \sum_{t=2}^T \| \mathbf{y}_{t, :4} - \mathbf{y}_{t-1, :4} \|_2^2$$

### 2.4 Loss Weight Hyperparameters
- $\alpha = 1.0$ (Mode classification)
- $\beta = 3.5$ (Steering precision — heavy weight to ensure smooth path tracking)
- $\gamma = 0.1$ (Temporal smoothness)

---

## 3. Sequence Handling & Recurrent BPTT

The GRU must learn temporal momentum and turn recovery:
1. **Window Size:** Sequences are cut into chunks of length $T = 64$ frames ($\approx 4.27$ seconds at 15 Hz).
2. **Hidden State Initialization:** At episode start ($t = 0$), $\mathbf{h}_0 = \mathbf{0}$.
3. **Truncated BPTT:** For contiguous chunks within the same episode, $\mathbf{h}_{\text{end}}$ from chunk $k$ is detached and passed as $\mathbf{h}_0$ for chunk $k+1$.
4. **Batch Organization:** Batches of 32 sequences (2,048 frames total) sampled across independent match episodes.

---

## 4. Optimization Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` | Standard adaptive gradient with decoupled weight decay. |
| **Learning Rate** | $3 \times 10^{-4}$ | Safe learning rate for small GRUs without gradient explosion. |
| **LR Scheduler** | `CosineAnnealingLR` ($T_{\max}=25, \eta_{\min}=10^{-5}$) | Smooth decay for stable convergence. |
| **Weight Decay** | $10^{-4}$ | Prevents weight magnitude explosion. |
| **Gradient Clipping** | $\text{clip\_norm} = 1.0$ | Protects GRU recurrent weights from gradient spikes. |
| **Epochs** | 25 | Sufficient for 50,000–100,000 frame demonstration sets. |
| **Early Stopping** | Patience = 5 epochs | Monitored on held-out validation loss. |

---

## 5. Protection of Production Submission and Artifacts

1. **Output Checkpoint:** Distillation training will output strictly to:
   `experiments/checkpoints/candidate_D_distilled.pt`
2. **Untouched Production Model:** `model/V13.pt` **must not be modified, overwritten, or linked** during training.
3. **PIN & Submission Untouched:** `src/runtime.py` PIN (`bf434bfe...`) and `submission.json` remain unchanged until Candidate D passes the Stage E Evaluation Protocol.
4. **Preflight Testing:** Candidate D will undergo offline preflight and cosine similarity checks in `experiments/` before any promotion consideration.

---

## 6. Execution Gate
> [!IMPORTANT]
> **GATE NOTICE:** Training is strictly disabled in Stage A.  
> Do not execute training scripts until Stage B (Collector Tooling) and Stage C (Demonstration Gathering) have been audited and approved `[REQUIRED APPROVAL]`.
