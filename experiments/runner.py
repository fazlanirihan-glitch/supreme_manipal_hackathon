"""Iterative Experiment Runner for Model Wars.
Trains and compares multiple candidate configurations, records metrics,
runs objective evaluation benchmarks, and selects the strongest candidate.
"""
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from policy import Policy
from evaluator import evaluate_checkpoint

DATASET_PATH = ROOT / 'experiments' / 'datasets' / 'dataset_v1.pt'
CHECKPOINTS_DIR = ROOT / 'experiments' / 'checkpoints'
LOG_PATH = ROOT / 'experiments' / 'experiment_log.jsonl'
BASELINE_PATH = ROOT / 'experiments' / 'baseline' / 'V13_baseline.pt'


def load_dataset():
    data = torch.load(DATASET_PATH, map_location='cpu', weights_only=False)
    episodes = data['episodes']
    
    # Flatten into sequences of length 64
    seq_len = 64
    all_obs, all_mode, all_steer = [], [], []
    for ep in episodes:
        for f in ep['frames']:
            all_obs.append(f['obs'])
            all_mode.append(f['mode'])
            all_steer.append(f['steer'])
            
    num_seqs = len(all_obs) // seq_len
    X = torch.tensor(all_obs[:num_seqs * seq_len], dtype=torch.float32).view(num_seqs, seq_len, 13)
    Y_mode = torch.tensor(all_mode[:num_seqs * seq_len], dtype=torch.long).view(num_seqs, seq_len)
    Y_steer = torch.tensor(all_steer[:num_seqs * seq_len], dtype=torch.float32).view(num_seqs, seq_len)
    
    # 80/20 chronological split
    split = int(0.8 * num_seqs)
    return (X[:split], Y_mode[:split], Y_steer[:split]), (X[split:], Y_mode[split:], Y_steer[split:])


def train_candidate(name, lr, steer_weight, class_weighted, epochs=12, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_data, val_data = load_dataset()
    train_X, train_mode, train_steer = train_data
    val_X, val_mode, val_steer = val_data
    
    # Mode class weights
    if class_weighted:
        mode_counts = [max((train_mode == c).sum().item(), 1) for c in range(4)]
        total = sum(mode_counts)
        weights = torch.tensor([total / (4.0 * c) for c in mode_counts], dtype=torch.float32)
        weights = weights / weights.mean()
        criterion_mode = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion_mode = nn.CrossEntropyLoss()
        
    criterion_steer = nn.SmoothL1Loss()
    
    # Model init from baseline
    model = Policy()
    baseline_saved = torch.load(BASELINE_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(baseline_saved['state_dict'], strict=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = -1
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for i in range(train_X.shape[0]):
            optimizer.zero_grad()
            logits, steer, _ = model(train_X[i], None)
            loss = criterion_mode(logits, train_mode[i]) + steer_weight * criterion_steer(steer, train_steer[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        steer_diffs = []
        with torch.no_grad():
            for i in range(val_X.shape[0]):
                logits, steer, _ = model(val_X[i], None)
                loss = criterion_mode(logits, val_mode[i]) + steer_weight * criterion_steer(steer, val_steer[i])
                val_losses.append(loss.item())
                correct += (logits.argmax(dim=-1) == val_mode[i]).sum().item()
                total += val_mode[i].numel()
                steer_diffs.append((steer - val_steer[i]).abs().mean().item())
                
        avg_val_loss = float(np.mean(val_losses))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    val_acc = correct / max(total, 1)
    val_steer_mae = float(np.mean(steer_diffs))
    
    # Save checkpoint
    ckpt_path = CHECKPOINTS_DIR / f"{name}.pt"
    ckpt_dict = {
        'format': 'v13_student_checkpoint_v1',
        'architecture': '13D / encoder64 / GRU64 / mode4+steer',
        'parameters': 26181,
        'state_dict': best_state_dict,
        'dataset_manifest_sha256': baseline_saved.get('dataset_manifest_sha256', '')
    }
    torch.save(ckpt_dict, ckpt_path)
    
    # Cosine similarity
    v_sub = torch.cat([p.flatten() for p in best_state_dict.values()])
    v_start = torch.cat([p.flatten() for p in baseline_saved['state_dict'].values()])
    cos_sim = torch.nn.functional.cosine_similarity(v_sub, v_start, dim=0).item()
    
    # Benchmark evaluation
    eval_metrics = evaluate_checkpoint(ckpt_path)
    
    result = {
        'experiment_name': name,
        'checkpoint_path': str(ckpt_path),
        'learning_rate': lr,
        'steer_weight': steer_weight,
        'class_weighted': class_weighted,
        'best_epoch': best_epoch,
        'best_val_loss': round(best_val_loss, 4),
        'val_mode_acc': round(val_acc, 4),
        'val_steer_mae': round(val_steer_mae, 4),
        'cosine_similarity': round(cos_sim, 5),
        'benchmarks': eval_metrics
    }
    
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result) + '\n')
        
    return result


def main():
    print("=" * 70)
    print("STARTING ITERATIVE EXPERIMENTATION SUITE")
    print("=" * 70)
    
    experiments = [
        # Candidate 1: Balanced weighted imitation
        {"name": "candidate_A_balanced", "lr": 3e-4, "steer_weight": 2.0, "class_weighted": True},
        # Candidate 2: High-precision steering regression
        {"name": "candidate_B_steer_heavy", "lr": 1.5e-4, "steer_weight": 4.0, "class_weighted": False},
        # Candidate 3: Fast learning rate with class weighting
        {"name": "candidate_C_rapid_adapt", "lr": 5e-4, "steer_weight": 2.5, "class_weighted": True},
    ]
    
    results = []
    for exp in experiments:
        print(f"\n---> Running Experiment: {exp['name']} (lr={exp['lr']}, steer_w={exp['steer_weight']}, weighted={exp['class_weighted']})...")
        res = train_candidate(**exp)
        results.append(res)
        print(f"Results for {exp['name']}:")
        print(f"  Val Loss: {res['best_val_loss']}, Mode Acc: {res['val_mode_acc']:.2%}, Steer MAE: {res['val_steer_mae']}")
        print(f"  Cosine Sim: {res['cosine_similarity']} (<= 0.999 check: {res['cosine_similarity'] <= 0.999})")
        print(f"  Kickoff Stalls: {res['benchmarks']['kickoff_stalled_ticks']} (Baseline was 6)")
        print(f"  Turnaround Mode 0 Ticks: {res['benchmarks']['turnaround_mode0_ticks']} (Baseline was 30)")
        print(f"  High Ball Front Dodges: {res['benchmarks']['high_ball_ground_front_dodges']} (Baseline was 2)")
        
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 70)
    
    # Sort candidates by combined score: lower turnaround stall + lower val loss + lower steer MAE
    for r in results:
        print(f"Candidate: {r['experiment_name']:<25} Val Loss: {r['best_val_loss']:<8} Mode Acc: {r['val_mode_acc']:<8.2%} Steer MAE: {r['val_steer_mae']:<8} Kickoff Stalls: {r['benchmarks']['kickoff_stalled_ticks']:<4} Turnaround M0: {r['benchmarks']['turnaround_mode0_ticks']:<4}")

    # Best selection
    best_candidate = min(results, key=lambda r: (r['benchmarks']['kickoff_stalled_ticks'], r['benchmarks']['turnaround_mode0_ticks'], r['best_val_loss']))
    print(f"\n[WINNER] Selected Best Candidate: {best_candidate['experiment_name']}")
    print(f"Checkpoint: {best_candidate['checkpoint_path']}")


if __name__ == '__main__':
    main()
