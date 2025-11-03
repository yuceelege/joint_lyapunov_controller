import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import math
import matplotlib.pyplot as plt
from map import Ring, Map
from agent import Drone
from environment import Environment
from controller import SimpleControllerNN
from lyap import LyapunovNet
from utils import *
from sampler import *
from propogate_visualize import *
from sublevel_sampler import *
import argparse

from config import DEVICE, DTYPE
DT = 0.1

model = SimpleControllerNN(color=True, image_size=50).to(DEVICE)
model.load_state_dict(torch.load('../../weights/joint_controller.pth', map_location=DEVICE))
model.eval()

net = LyapunovNet(
    state_dim=7,
    xi_star=sample_sublevel(
        1, 0,
        torch.tensor([0.,0.,1.], device=DEVICE, dtype=DTYPE),
        torch.tensor([1.,0.,0.], device=DEVICE, dtype=DTYPE)
    ),
    epsilon=1e-3
).to(DEVICE)
net.load_state_dict(torch.load('../../weights/joint_lyap.pth', map_location=DEVICE))
net.eval()

def run_test(num_samples):
    center2 = torch.tensor([0.0,0.,0.], device=DEVICE, dtype=DTYPE)
    normal  = torch.tensor([1.,0.,0.], device=DEVICE, dtype=DTYPE)
    center   = torch.tensor([0.0,0.,0.], device=DEVICE, dtype=DTYPE)
    r_in, r_out = 0.5, 1.0
    samples, _, _ = sample_candidate_S(num_samples, center2, normal)
    decrease_and_inside = 0
    # increase = 0
    # outside = 0
    bad_case = 0
    r_good = []
    phi_good = []
    r_bad = []
    phi_bad = []

    for s0 in samples:
        state0 = sample_to_drone_state(s0, center2)
        if is_inside_gate_region(center2, normal, state0, r_in, r_out):
            continue
        V0 = net(s0.unsqueeze(0).to(DEVICE)).cpu().item()
        s1 = one_step(s0, center, model)
        state1 = sample_to_drone_state(s1, center2)
        V1 = net(s1.unsqueeze(0).to(DEVICE)).cpu().item()
        inside_B = is_interior(s1)
        in_gate    = is_inside_gate_region(center2, normal, state1, r_in, r_out)
        r0   = s0[0].item()
        phi0 = s0[1].item()
        if in_gate:
            decrease_and_inside += 1
            r_good.append(r0)
            phi_good.append(phi0)
        elif V1<V0 and inside_B:
            decrease_and_inside += 1
            r_good.append(r0)
            phi_good.append(phi0)
        else:
            bad_case +=1
            r_bad.append(r0)
            phi_bad.append(phi0)

    print(decrease_and_inside, bad_case)
    phi_bad  = [p + math.pi for p in phi_bad]
    phi_good = [p + math.pi for p in phi_good]
    plt.figure()
    ax = plt.subplot(projection='polar')
    ax.scatter(phi_bad,  r_bad,  s=10, alpha=0.6, c='red')
    ax.scatter(phi_good, r_good, s=1, alpha=0.1, c='green')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=3000)
    args = parser.parse_args()
    run_test(args.num_samples)
