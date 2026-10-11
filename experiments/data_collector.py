"""Sequential Episode Data Collector for Model Wars.
Generates contiguous, episode-aware match trajectories with exact 13D observations
and synchronized expert action labels from GoalAwareExpert.
"""
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))

import torch
from observation_contract import basis, build
from expert_policy import GoalAwareExpert

DT = 1.0 / 60.0 # 60 Hz nominal step


def simulate_ball_step(ball_pos, ball_vel):
    """Simulates 1 tick of ball physics on standard pitch."""
    bx, by, bz = ball_pos
    vx, vy, vz = ball_vel

    # Gravity
    vz -= 650.0 * DT

    # Drag
    drag = 0.998
    vx *= drag
    vy *= drag
    vz *= drag

    # Position integration
    bx += vx * DT
    by += vy * DT
    bz += vz * DT

    # Floor collision
    if bz <= 93.0:
        bz = 93.0
        vz = -vz * 0.60 # Elastic rebound
        vx *= 0.95
        vy *= 0.95

    # Side wall collisions (X = +/- 4096)
    if abs(bx) >= 4000.0:
        bx = 4000.0 if bx > 0 else -4000.0
        vx = -vx * 0.60

    # Back wall collisions (Y = +/- 5120)
    if abs(by) >= 5050.0:
        # Check goal mouth (X within 893, Z within 642)
        if abs(bx) > 893.0 or bz > 642.0:
            by = 5050.0 if by > 0 else -5050.0
            vy = -vy * 0.60

    return [bx, by, bz], [vx, vy, vz]


def simulate_car_step(car_pos, car_vel, car_pyr, throttle, steer, mode):
    """Simulates 1 tick of car ground kinematics."""
    cx, cy, cz = car_pos
    vx, vy, vz = car_vel
    pitch, yaw, roll = car_pyr

    speed = math.sqrt(vx**2 + vy**2 + vz**2)

    # Throttle acceleration
    if throttle > 0:
        accel = 1600.0 if speed < 1400.0 else (1000.0 if speed < 2200.0 else 0.0)
    else:
        accel = -525.0 # Coasting drag

    # Heading update via Ackermann steering curvature
    if speed > 10.0:
        # Turning radius decreases with speed
        curvature = 0.0069 - 0.000398 * (speed / 100.0)
        curvature = max(curvature, 0.0015)
        angular_vel = steer * speed * curvature
        yaw += angular_vel * DT

    # Forward direction in world frame
    fwd_x = math.cos(pitch) * math.cos(yaw)
    fwd_y = math.cos(pitch) * math.sin(yaw)

    new_speed = max(0.0, min(2300.0, speed + accel * DT))
    vx = fwd_x * new_speed
    vy = fwd_y * new_speed

    cx += vx * DT
    cy += vy * DT

    # Arena boundary clamps
    cx = max(-4000.0, min(4000.0, cx))
    cy = max(-5050.0, min(5050.0, cy))
    cz = 17.0 # Ground plane

    return [cx, cy, cz], [vx, vy, vz], [pitch, yaw, roll]


def collect_dataset(num_episodes=30, episode_len=240, seed=42):
    """
    Collects a balanced, sequential dataset of match episodes.
    Total frames = num_episodes * episode_len.
    """
    random.seed(seed)
    expert = GoalAwareExpert(team=0)
    episodes = []

    scenario_types = ['kickoff', 'offensive_attack', 'defensive_clear', 'turnaround', 'high_aerial']

    for ep_id in range(num_episodes):
        scenario = scenario_types[ep_id % len(scenario_types)]
        expert.reset()

        # Initialize scenario state
        if scenario == 'kickoff':
            # Kickoff positions
            car_pos = [0.0, -4608.0, 17.0]
            car_pyr = [0.0, math.pi / 2.0, 0.0] # Facing +Y
            car_vel = [0.0, 0.0, 0.0]
            ball_pos = [0.0, 0.0, 93.0]
            ball_vel = [0.0, 0.0, 0.0]
        elif scenario == 'offensive_attack':
            car_pos = [random.uniform(-1500, 1500), random.uniform(-1000, 1000), 17.0]
            car_pyr = [0.0, math.pi / 2.0 + random.uniform(-0.3, 0.3), 0.0]
            car_vel = [random.uniform(-200, 200), random.uniform(500, 1200), 0.0]
            ball_pos = [random.uniform(-1500, 1500), random.uniform(1500, 3500), 93.0]
            ball_vel = [random.uniform(-300, 300), random.uniform(-200, 400), 0.0]
        elif scenario == 'defensive_clear':
            # Ball in our defensive half, car needs arcing approach
            car_pos = [random.uniform(-1000, 1000), random.uniform(-3500, -2000), 17.0]
            car_pyr = [0.0, -math.pi / 2.0, 0.0] # Heading toward own net
            car_vel = [0.0, -500.0, 0.0]
            ball_pos = [random.uniform(-1200, 1200), random.uniform(-4200, -3000), 93.0]
            ball_vel = [random.uniform(-200, 200), random.uniform(-400, -100), 0.0]
        elif scenario == 'turnaround':
            # Ball behind car
            car_pos = [random.uniform(-1500, 1500), random.uniform(-1000, 1000), 17.0]
            car_pyr = [0.0, math.pi / 2.0, 0.0] # Facing +Y
            car_vel = [0.0, 600.0, 0.0]
            ball_pos = [car_pos[0] + random.uniform(-1000, 1000), car_pos[1] - random.uniform(800, 2000), 93.0]
            ball_vel = [random.uniform(-200, 200), random.uniform(-200, 200), 0.0]
        else: # high_aerial
            car_pos = [random.uniform(-1500, 1500), random.uniform(-1500, 1500), 17.0]
            car_pyr = [0.0, random.uniform(-math.pi, math.pi), 0.0]
            car_vel = [0.0, 500.0, 0.0]
            ball_pos = [car_pos[0] + random.uniform(-400, 400), car_pos[1] + random.uniform(200, 800), random.uniform(800, 1600)]
            ball_vel = [random.uniform(-200, 200), random.uniform(-200, 200), random.uniform(200, 600)]

        ep_frames = []
        prev_elapsed = None
        prev_controls = None

        for t in range(episode_len):
            elapsed = t * DT

            # Compute predicted ball slice at T + 2.0s
            pred_slice_pos = [
                ball_pos[0] + ball_vel[0] * 2.0,
                ball_pos[1] + ball_vel[1] * 2.0,
                max(93.0, ball_pos[2] + ball_vel[2] * 2.0 - 0.5 * 650.0 * 4.0)
            ]

            # Query expert action
            mode, steer = expert.get_action(
                car_pos, car_vel, car_pyr, ball_pos, ball_vel, pred_slice_pos
            )

            # Build exact 13D observation using observation_contract
            prior_mode = 0 if prev_controls is None else (
                3 if prev_controls[5] and prev_controls[2] == -1 else
                2 if prev_controls[5] else
                1 if prev_controls[0] == 1 else 0
            )
            context = dict(
                elapsed=elapsed,
                previous_elapsed=prev_elapsed,
                previous_mode=prior_mode,
                previous_steer=0.0 if prev_controls is None or prior_mode != 1 else prev_controls[1],
                prediction_valid=True,
                prediction_position=pred_slice_pos,
                first_time=elapsed,
                selected_time=elapsed + 2.0
            )
            raw = dict(
                position=car_pos,
                velocity=car_vel,
                basis=basis(car_pyr),
                ball_present=True,
                ball_position=ball_pos
            )
            obs13 = list(build(raw, context))[:13]

            ep_frames.append({
                'obs': obs13,
                'mode': mode,
                'steer': steer,
                'T': elapsed,
                'scenario': scenario
            })

            # Decode controls for car simulation step
            throttle = 1.0 if mode == 1 else 0.0
            car_pos, car_vel, car_pyr = simulate_car_step(car_pos, car_vel, car_pyr, throttle, steer, mode)
            ball_pos, ball_vel = simulate_ball_step(ball_pos, ball_vel)

            prev_elapsed = elapsed
            prev_controls = [throttle, steer, -1.0 if mode == 3 else 0.0, 0.0, 0.0, mode in (2, 3), False, False]

        episodes.append({
            'episode_id': ep_id,
            'scenario': scenario,
            'frames': ep_frames
        })

    return episodes
