import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import math
from map import Map, Ring
from agent import Drone
from environment import Environment
from controller import SimpleControllerNN
from utils import *

from config import DEVICE, DTYPE
DT = 0.1

# model = SimpleControllerNN(color=True, image_size=50).to(DEVICE)
# model.load_state_dict(torch.load('../../weights/dagger_weights.pth', map_location=DEVICE))
# model.eval()

def sample_to_drone_state(sample, center):
    r_val, phi, z_val, yaw, dr, dz, dyaw = sample.unbind(0)
    x = center[0] + r_val * torch.cos(phi)
    y = center[1] + r_val * torch.sin(phi)
    z = center[2] + z_val
    state6 = torch.zeros(7, device=DEVICE, dtype=DTYPE)
    state6[0] = x
    state6[1] = y
    state6[2] = z
    state6[3] = dr
    state6[4] = dz
    state6[5] = yaw
    state6[6] = dyaw
    return state6

def drone_state_to_sample(state6, center):
    x, y, z_w, dr, dz, yaw, dyaw = state6.unbind(0)
    dx = x - center[0]
    dy = y - center[1]
    eps = 1e-9

    r_val = torch.sqrt((dx*dx + dy*dy).clamp_min(eps))

    # compute raw φ in (-π,π]
    phi0 = torch.atan2(dy, dx + eps)

    # remap into [0,2π) so no jump at +π/-π
    TWO_PI = 2 * math.pi
    phi    = torch.remainder(phi0 + TWO_PI, TWO_PI)

    return torch.stack([r_val,
                        phi,
                        z_w - center[2],
                        yaw,
                        dr,
                        dz,
                        dyaw])

def one_step(sample, center, model):
    state7 = sample_to_drone_state(sample, center)
    agent = Drone(state7)

    ring_center = torch.tensor([0.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    ring_normal = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    ring = Ring(ring_center.cpu().numpy(), ring_normal.cpu().numpy(), inner_radius=0.5, outer_radius=1.0)
    mp = Map(rings=[ring])

    s6 = agent.get_state().unsqueeze(0)
    yaw = s6[0, 5]
    euler = torch.tensor([0.0, 0.0, yaw], device=DEVICE, dtype=DTYPE)
    R_wc = euler_to_rot_matrix(*euler)
    R_cw = R_wc.T.unsqueeze(0)

    env = Environment(agent, mp)
    img = env.renderBatch(s6, R_cw)[0]
    obs = img.permute(2, 0, 1).unsqueeze(0)

    r = s6[0, 3]
    vz = s6[0, 4]
    th = s6[0, 5]
    yaw_rate = s6[0,6]
    lin_vel = torch.stack([
                r * torch.cos(th),
                r * torch.sin(th),
                vz,
                yaw_rate
            ], dim=0)            # → a 4-vector now
    vel_in = lin_vel.unsqueeze(0)
    pos = s6[0, :3]
    rel_to_gate = (ring_center - pos).unsqueeze(0)

   # with torch.no_grad():
    u_policy = model(obs.to(DEVICE), vel_in.to(DEVICE), rel_to_gate.to(DEVICE)).squeeze(0)

    agent.update_state(u_policy, DT)
    updated = agent.get_state()
    return drone_state_to_sample(updated, center)

