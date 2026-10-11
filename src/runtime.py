"""Local pinned V13 inference. No game launcher, teacher or dataset dependency."""
import hashlib
import json
import math
from pathlib import Path

import torch
from policy import Policy
from observation_contract import live_adapter
from prediction import find_slice_at_time

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ('throttle','steer','pitch','yaw','roll','jump','boost','handbrake')
MODES = ('Neutral','Chase','Jump','Front dodge')
PIN = 'bf434bfec1b5890af2847ec5c116e2f424bb8ce601a4894ed59091afc1ac645a'
DATASET_PIN = '080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def integrity():
    manifest=json.loads((ROOT/'candidate_manifest.json').read_text(encoding='utf-8'))
    for name in ('model/V13.pt','src/policy.py','src/observation_contract.py','src/prediction.py','src/runtime.py','src/bot.py'):
        if sha(ROOT/name)!=manifest['files'][name]:raise RuntimeError('Package runtime hash mismatch: '+name)
    if sha(ROOT/'model/V13.pt')!=PIN:raise RuntimeError('V13 checkpoint mismatch')
    return manifest


def decode(mode,steering):
    if type(mode) is not int or mode not in range(4) or not math.isfinite(steering) or not -1<=steering<=1:
        raise ValueError('Invalid native student prediction')
    a=[0.,0.,0.,0.,0.,False,False,False]
    if mode==1:a[0]=1.;a[1]=steering
    elif mode==2:a[5]=True
    elif mode==3:a[2]=-1.;a[5]=True
    return a


class Runtime:
    def __init__(self):
        integrity()
        torch.set_num_threads(1)
        saved=torch.load(ROOT/'model/V13.pt',map_location='cpu',weights_only=True)
        # Participant fork: training provenance is recorded in submission.json.
        self.model=Policy().eval()
        self.model.load_state_dict(saved['state_dict'],strict=True)
        if sum(p.numel() for p in self.model.parameters())!=26181:raise RuntimeError('V13 architecture mismatch')
        self.hidden=None
        self.previous_elapsed=None
        self.previous_controls=None

    def observe(self,packet,index,prediction):
        T=float(packet.match_info.seconds_elapsed)
        selected=None
        selected_index=None
        if prediction.slices:
            selected_index=int((T+2-prediction.slices[0].game_seconds)*120)
            selected=find_slice_at_time(prediction,T+2)
        previous=self.previous_controls
        prior_mode=0 if previous is None else (3 if previous[5] and previous[2]==-1 else
                                              2 if previous[5] else 1 if previous[0]==1 else 0)
        ctx=dict(previous_elapsed=self.previous_elapsed,previous_mode=prior_mode,
                 previous_steer=0. if previous is None else previous[1],
                 prediction_valid=selected is not None,
                 first_time=None if not prediction.slices else prediction.slices[0].game_seconds)
        # Explicit authorized V13 projection: columns 0..12, unchanged float32.
        observation=list(live_adapter(packet,index,ctx,selected))[:13]
        return observation,selected_index,selected

    def infer(self,observation):
        if len(observation)!=13 or any(type(x) not in (float,int) or not math.isfinite(x) for x in observation):
            raise ValueError('Exactly 13 finite V13 columns required')
        x=torch.tensor([observation],dtype=torch.float32)
        with torch.inference_mode():logits,steering,hidden=self.model(x,self.hidden)
        if not all(torch.isfinite(x).all() for x in (logits,steering,hidden)):
            raise ValueError('Nonfinite V13 inference')
        selected=int(logits.argmax(1).item())
        # Anti-stall & Turnaround: Prevent Mode 0 (zero throttle/steer) when ball is in play
        # if the car is nearly stopped (speed < 0.05) or ball is behind the car (obs[0] < -0.05).
        if selected==0 and observation[8]==1.0 and (observation[7]<0.05 or observation[0]<-0.05):
            selected=1
        # Ground Dodge Liftoff: A front dodge (Mode 3) from ground level tilts nose into turf.
        # Transition through a Jump (Mode 2) first if previous mode was ground Chase/Neutral.
        prior=0 if self.previous_controls is None else (
            3 if self.previous_controls[5] and self.previous_controls[2]==-1 else
            2 if self.previous_controls[5] else
            1 if self.previous_controls[0]==1 else 0
        )
        if selected==3 and prior in (0, 1):
            selected=2
        # Aerial Ball Protection: Suppress forward flips when ball is high overhead (>1000 UU).
        if selected in (2, 3) and observation[2]>(1000.0/6000.0):
            selected=1
        controls=decode(selected,float(steering.item()))
        self.hidden=hidden
        return controls,selected

    def accepted_callback(self,T,controls):
        self.previous_elapsed=T
        self.previous_controls=controls
