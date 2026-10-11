"""Live RLBot 60Hz/120Hz trajectory recorder and reference teacher bot.
Records exact 13D observations and synchronized teacher action labels (mode, steer).
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference' / 'python_example' / 'src'))
sys.path.insert(0, str(ROOT / 'src'))

import torch
from rlbot.flat import ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot
from observation_contract import live_adapter
from prediction import find_slice_at_time
from util.drive import steer_toward_target
from util.sequence import ControlStep, Sequence
from util.vec import Vec3

DATASET_PATH = ROOT / 'tools' / 'dataset_live.pt'


class LiveRecorderBot(Bot):
    def initialize(self):
        self.active_sequence = None
        self.frames = []
        self.previous_elapsed = None
        self.previous_controls = None
        self.saved_count = 0
        print(f"[Recorder] Initialized! Will record live frames to: {DATASET_PATH}", flush=True)

    def get_output(self, packet: GamePacket) -> ControllerState:
        if len(packet.balls) == 0:
            return ControllerState()

        T = float(packet.match_info.seconds_elapsed)
        my_car = packet.players[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        speed = car_velocity.length()
        ball_location = Vec3(packet.balls[0].physics.location)

        # 1. Determine teacher target location
        target_location = ball_location
        selected_pred = None
        if self.ball_prediction and self.ball_prediction.slices:
            selected_pred = find_slice_at_time(self.ball_prediction, T + 2)
            if car_location.dist(ball_location) > 1500 and selected_pred is not None:
                target_location = Vec3(selected_pred.physics.location)

        # 2. Determine teacher action and mode
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            # Map sequence controls to mode
            if controls.jump and controls.pitch == -1.0:
                mode = 3 # Front dodge
            elif controls.jump:
                mode = 2 # Jump
            elif controls.throttle == 1.0:
                mode = 1 # Chase
            else:
                mode = 0 # Neutral / Coast
            steer = float(controls.steer)
        elif 750 < speed < 800:
            # Begin front flip
            self.active_sequence = Sequence([
                ControlStep(duration=0.05, controls=ControllerState(jump=True)),
                ControlStep(duration=0.05, controls=ControllerState(jump=False)),
                ControlStep(duration=0.2, controls=ControllerState(jump=True, pitch=-1)),
                ControlStep(duration=0.8, controls=ControllerState()),
            ])
            controls = self.active_sequence.tick(packet)
            mode = 2
            steer = 0.0
        else:
            steer_val = steer_toward_target(my_car, target_location)
            controls = ControllerState(throttle=1.0, steer=steer_val)
            mode = 1
            steer = float(steer_val)

        # 3. Build exact 13D observation using authorized live_adapter
        prev = self.previous_controls
        prior_mode = 0 if prev is None else (
            3 if prev[5] and prev[2] == -1 else
            2 if prev[5] else
            1 if prev[0] == 1 else 0
        )
        ctx = dict(
            previous_elapsed=self.previous_elapsed,
            previous_mode=prior_mode,
            previous_steer=0.0 if prev is None else prev[1],
            prediction_valid=selected_pred is not None,
            first_time=None if not (self.ball_prediction and self.ball_prediction.slices) else self.ball_prediction.slices[0].game_seconds,
            selected_time=None if selected_pred is None else selected_pred.game_seconds
        )
        obs = list(live_adapter(packet, self.index, ctx, selected_pred))[:13]

        # 4. Record frame during active/kickoff phases
        phase = packet.match_info.match_phase
        if phase in (MatchPhase.Active, MatchPhase.Kickoff) and len(obs) == 13 and all(math.isfinite(x) for x in obs):
            self.frames.append({
                'obs': obs,
                'mode': int(mode),
                'steer': float(steer),
                'T': T
            })
            if len(self.frames) % 300 == 0:
                self.save_snapshot()

        self.previous_elapsed = T
        self.previous_controls = [controls.throttle, controls.steer, controls.pitch, controls.yaw, controls.roll, controls.jump, controls.boost, controls.handbrake]
        return controls

    def save_snapshot(self):
        torch.save({
            'count': len(self.frames),
            'frames': self.frames,
            'source': 'live_RLBot_reference_teacher',
            'agent_id': 'rlbot_community/python_example'
        }, DATASET_PATH)
        print(f"[Recorder] Saved snapshot: {len(self.frames)} frames ({len(self.frames)/60:.1f}s active play)", flush=True)

    def retire(self):
        self.save_snapshot()
        print(f"[Recorder] Match retired! Total frames recorded: {len(self.frames)}", flush=True)
        super().retire()


if __name__ == '__main__':
    LiveRecorderBot('rlbot_community/python_example').run(wants_ball_predictions=True, wants_match_communications=False)
