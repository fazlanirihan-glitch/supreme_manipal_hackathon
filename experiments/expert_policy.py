"""Goal-Aware Defensive & Interception Expert Policy for Model Wars.
Generates genuine expert actions that avoid own-goal trajectories,
execute arcing clearances, maintain behind-the-car turnarounds,
and time front flips properly.
"""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from observation_contract import basis


def clamp(val, low, high):
    return max(low, min(high, val))


def relative_location(car_pos, car_pyr, target_pos):
    """Transform target_pos into car local coordinates (forward, right, up)."""
    p, y, r = car_pyr
    cp, cy, cr = math.cos(p), math.cos(y), math.cos(r)
    sp, sy, sr = math.sin(p), math.sin(y), math.sin(r)
    axes = (
        (cp * cy, cp * sy, sp),                                       # Forward
        (cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr),   # Right
        (-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr)   # Up
    )
    delta = [t - c for t, c in zip(target_pos, car_pos)]
    rel_x = sum(d * a for d, a in zip(delta, axes[0]))
    rel_y = sum(d * a for d, a in zip(delta, axes[1]))
    rel_z = sum(d * a for d, a in zip(delta, axes[2]))
    return rel_x, rel_y, rel_z


def steer_toward_target(car_pos, car_pyr, target_pos):
    """Calculates continuous steering [-1, 1] toward target."""
    rel_x, rel_y, _ = relative_location(car_pos, car_pyr, target_pos)
    angle = math.atan2(rel_y, rel_x)
    return clamp(angle * 5.0, -1.0, 1.0)


class GoalAwareExpert:
    """Expert driver that incorporates field geometry and own-goal protection."""
    def __init__(self, team=0):
        self.team = team # 0 = Blue (defends -Y, attacks +Y), 1 = Orange (defends +Y, attacks -Y)
        self.flip_tick = 0
        self.is_flipping = False

    def reset(self):
        self.flip_tick = 0
        self.is_flipping = False

    def get_action(self, car_pos, car_vel, car_pyr, ball_pos, ball_vel, pred_slice_pos=None):
        """
        Computes (mode, steer) action for current state.
        Modes: 0=Neutral, 1=Chase, 2=Jump, 3=Front dodge.
        Steer: float in [-1.0, 1.0].
        """
        speed = math.sqrt(sum(v**2 for v in car_vel))
        car_x, car_y, car_z = car_pos
        ball_x, ball_y, ball_z = ball_pos

        # Check flip sequence in progress
        if self.is_flipping:
            self.flip_tick += 1
            # 60 Hz sequence:
            # Ticks 1-3 (~0.05s): Mode 2 (Jump)
            # Ticks 4-6 (~0.05s): Mode 0 (Neutral liftoff)
            # Ticks 7-18 (~0.20s): Mode 3 (Front Dodge with pitch=-1)
            # Ticks 19-66 (~0.80s): Mode 0 (Neutral landing)
            if self.flip_tick <= 3:
                return 2, 0.0
            elif self.flip_tick <= 6:
                return 0, 0.0
            elif self.flip_tick <= 18:
                return 3, 0.0
            elif self.flip_tick <= 66:
                return 0, 0.0
            else:
                self.is_flipping = False
                self.flip_tick = 0

        # Defensive Analysis & Target Calculation
        own_goal_y = -5120.0 if self.team == 0 else 5120.0
        opp_goal_y = 5120.0 if self.team == 0 else -5120.0
        attack_direction = 1.0 if self.team == 0 else -1.0

        # Is ball on defensive half?
        in_defensive_half = (ball_y * attack_direction) < 0.0

        # Is car facing own goal while between ball and own net?
        # Danger condition: car is between ball and our own goal
        car_is_wrong_side = (car_y * attack_direction) > (ball_y * attack_direction)

        if in_defensive_half and car_is_wrong_side:
            # DANGER OF OWN GOAL: If we drive straight toward the ball, we push it into our net!
            # Instead: swing wide around the ball to approach from behind toward the side wall
            side_sign = 1.0 if car_x >= 0 else -1.0
            arc_x = ball_x + side_sign * 900.0 # Swing wide
            arc_y = ball_y - attack_direction * 600.0 # Swing behind ball
            target_pos = [arc_x, arc_y, 17.0]
        else:
            # Safe attacking or interception
            dist_to_ball = math.sqrt(sum((b - c)**2 for b, c in zip(ball_pos, car_pos)))
            if dist_to_ball > 1500.0 and pred_slice_pos is not None:
                target_pos = list(pred_slice_pos)
            else:
                target_pos = list(ball_pos)

        # Steering calculation toward selected target
        steer = steer_toward_target(car_pos, car_pyr, target_pos)

        # Check if front flip is appropriate:
        # 1. Car moving at high speed (750 < v < 850)
        # 2. Car roughly aligned with target (abs(steer) < 0.2)
        # 3. Ball is NOT high in the air (ball_z < 600 UU)
        # 4. Safe attacking situation (not wrong side of own net)
        rel_x, _, _ = relative_location(car_pos, car_pyr, target_pos)
        if (750.0 < speed < 850.0 and abs(steer) < 0.25 and 
            ball_z < 600.0 and rel_x > 800.0 and not car_is_wrong_side):
            self.is_flipping = True
            self.flip_tick = 1
            return 2, 0.0 # Begin with Jump liftoff

        # Default action: Mode 1 (Chase) with continuous steering
        return 1, float(steer)
