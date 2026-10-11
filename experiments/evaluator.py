"""Evaluation Benchmark Suite for Model Wars Candidates.
Runs standardized gameplay challenges on any policy checkpoint and produces
objective quantitative metrics for baseline and candidate comparison.
"""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

import torch
from policy import Policy
from observation_contract import basis, build
from data_collector import simulate_car_step, simulate_ball_step, DT


def run_kickoff_challenge(model, num_steps=120):
    """Measures kickoff acceleration and absence of Mode 0 deadlock from rest."""
    car_pos = [0.0, -4608.0, 17.0]
    car_vel = [0.0, 0.0, 0.0]
    car_pyr = [0.0, math.pi / 2.0, 0.0]
    ball_pos = [0.0, 0.0, 93.0]
    ball_vel = [0.0, 0.0, 0.0]

    h = None
    prev_elapsed = None
    prev_controls = None
    stalled_ticks = 0

    for t in range(num_steps):
        elapsed = t * DT
        prior_mode = 0 if prev_controls is None else (
            3 if prev_controls[5] and prev_controls[2] == -1 else
            2 if prev_controls[5] else
            1 if prev_controls[0] == 1 else 0
        )
        context = dict(
            elapsed=elapsed, previous_elapsed=prev_elapsed, previous_mode=prior_mode,
            previous_steer=0.0 if prev_controls is None or prior_mode != 1 else prev_controls[1],
            prediction_valid=True, prediction_position=ball_pos, first_time=elapsed, selected_time=elapsed + 2.0
        )
        raw = dict(position=car_pos, velocity=car_vel, basis=basis(car_pyr), ball_present=True, ball_position=ball_pos)
        obs13 = list(build(raw, context))[:13]

        with torch.no_grad():
            logits, steer_pred, h = model(torch.tensor([obs13], dtype=torch.float32), h)
        mode = logits.argmax(dim=-1).item()
        steer = float(steer_pred.item())

        if mode == 0:
            stalled_ticks += 1

        throttle = 1.0 if mode == 1 else 0.0
        car_pos, car_vel, car_pyr = simulate_car_step(car_pos, car_vel, car_pyr, throttle, steer, mode)
        prev_elapsed = elapsed
        prev_controls = [throttle, steer, -1.0 if mode == 3 else 0.0, 0.0, 0.0, mode in (2, 3), False, False]

    final_speed = math.sqrt(sum(v**2 for v in car_vel))
    dist_covered = car_pos[1] - (-4608.0)
    return {
        'kickoff_final_speed': round(final_speed, 1),
        'kickoff_distance': round(dist_covered, 1),
        'kickoff_stalled_ticks': stalled_ticks
    }


def run_turnaround_challenge(model, num_steps=120):
    """Measures ability to steer around when ball is behind the car."""
    car_pos = [0.0, 0.0, 17.0]
    car_vel = [0.0, 500.0, 0.0]
    car_pyr = [0.0, math.pi / 2.0, 0.0] # Heading +Y
    ball_pos = [0.0, -1500.0, 93.0] # Ball 1500 UU behind
    ball_vel = [0.0, 0.0, 0.0]

    h = None
    prev_elapsed = None
    prev_controls = None
    steer_sum = 0.0
    mode0_ticks = 0

    for t in range(num_steps):
        elapsed = t * DT
        prior_mode = 0 if prev_controls is None else (
            3 if prev_controls[5] and prev_controls[2] == -1 else
            2 if prev_controls[5] else
            1 if prev_controls[0] == 1 else 0
        )
        context = dict(
            elapsed=elapsed, previous_elapsed=prev_elapsed, previous_mode=prior_mode,
            previous_steer=0.0 if prev_controls is None or prior_mode != 1 else prev_controls[1],
            prediction_valid=True, prediction_position=ball_pos, first_time=elapsed, selected_time=elapsed + 2.0
        )
        raw = dict(position=car_pos, velocity=car_vel, basis=basis(car_pyr), ball_present=True, ball_position=ball_pos)
        obs13 = list(build(raw, context))[:13]

        with torch.no_grad():
            logits, steer_pred, h = model(torch.tensor([obs13], dtype=torch.float32), h)
        mode = logits.argmax(dim=-1).item()
        steer = float(steer_pred.item())

        steer_sum += abs(steer)
        if mode == 0:
            mode0_ticks += 1

        throttle = 1.0 if mode == 1 else 0.0
        car_pos, car_vel, car_pyr = simulate_car_step(car_pos, car_vel, car_pyr, throttle, steer, mode)
        prev_elapsed = elapsed
        prev_controls = [throttle, steer, -1.0 if mode == 3 else 0.0, 0.0, 0.0, mode in (2, 3), False, False]

    # Heading change: did it rotate away from +Y?
    heading_change = abs(car_pyr[1] - math.pi / 2.0)
    return {
        'turnaround_heading_change_deg': round(math.degrees(heading_change), 1),
        'turnaround_mode0_ticks': mode0_ticks,
        'turnaround_avg_steer': round(steer_sum / num_steps, 3)
    }


def run_aerial_challenge(model, num_steps=60):
    """Measures policy behavior when ball is 1500 UU high."""
    car_pos = [0.0, 0.0, 17.0]
    car_vel = [0.0, 700.0, 0.0]
    car_pyr = [0.0, math.pi / 2.0, 0.0]
    ball_pos = [0.0, 500.0, 1500.0] # 1500 UU in the air
    ball_vel = [0.0, 0.0, 0.0]

    h = None
    prev_elapsed = None
    prev_controls = None
    mode3_count = 0

    for t in range(num_steps):
        elapsed = t * DT
        prior_mode = 0 if prev_controls is None else (
            3 if prev_controls[5] and prev_controls[2] == -1 else
            2 if prev_controls[5] else
            1 if prev_controls[0] == 1 else 0
        )
        context = dict(
            elapsed=elapsed, previous_elapsed=prev_elapsed, previous_mode=prior_mode,
            previous_steer=0.0 if prev_controls is None or prior_mode != 1 else prev_controls[1],
            prediction_valid=True, prediction_position=ball_pos, first_time=elapsed, selected_time=elapsed + 2.0
        )
        raw = dict(position=car_pos, velocity=car_vel, basis=basis(car_pyr), ball_present=True, ball_position=ball_pos)
        obs13 = list(build(raw, context))[:13]

        with torch.no_grad():
            logits, steer_pred, h = model(torch.tensor([obs13], dtype=torch.float32), h)
        mode = logits.argmax(dim=-1).item()
        if mode == 3:
            mode3_count += 1

        throttle = 1.0 if mode == 1 else 0.0
        car_pos, car_vel, car_pyr = simulate_car_step(car_pos, car_vel, car_pyr, throttle, float(steer_pred.item()), mode)
        prev_elapsed = elapsed
        prev_controls = [throttle, float(steer_pred.item()), -1.0 if mode == 3 else 0.0, 0.0, 0.0, mode in (2, 3), False, False]

    return {
        'high_ball_ground_front_dodges': mode3_count
    }


def evaluate_checkpoint(checkpoint_path):
    """Evaluates a checkpoint path and returns full benchmark dictionary."""
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = Policy().eval()
    model.load_state_dict(saved['state_dict'], strict=True)

    metrics = {}
    metrics.update(run_kickoff_challenge(model))
    metrics.update(run_turnaround_challenge(model))
    metrics.update(run_aerial_challenge(model))
    return metrics
