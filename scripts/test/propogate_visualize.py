import sys
from pathlib import Path
# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from map import Map, Ring
from agent import Drone
from environment import Environment
from controller import SimpleControllerNN
from utils import *
import argparse
from sampler import *
import matplotlib.pyplot as plt
from config import DEVICE, DTYPE
DT = 0.1
model     = SimpleControllerNN(color=True, image_size=50).to(DEVICE)

model.load_state_dict(torch.load('../../weights/joint_controller.pth', map_location=DEVICE))
model.eval()

def sample_to_drone_state(sample, center):
    r_val, phi, z_val, yaw, dr, dz, dyaw = sample.unbind(0)
    phi = phi+torch.pi
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
    phi = phi-torch.pi
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
    print(drone_state_to_sample(updated, center))
    print(is_interior(drone_state_to_sample(updated, center)))
    return drone_state_to_sample(updated, center)

def run_test(v_ref=1.0,
             dt=0.1,
             steps_per_episode=200,
             pass_thresh=0.1,
             visualize=True,
             rot_noise_std=0.05):
    center = torch.tensor([0.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    normal = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    sample_batch, _, _ = sample_interior_points(1, center, normal)
    sample = sample_batch[0]
    r_val, phi, z_val, yaw, dr, dz, dyaw = sample.unbind(0)
    x = center[0] + r_val * torch.cos(phi+torch.pi)
    y = center[1] + r_val * torch.sin(phi+torch.pi)
    z = center[2] + z_val

    init_state = torch.zeros(7, device=DEVICE, dtype=DTYPE)
    init_state[0] = x
    init_state[1] = y
    init_state[2] = z
    init_state[3] = dr
    init_state[4] = dz
    init_state[5] = yaw
    init_state[6] = dyaw

    ring = Ring(center.cpu().numpy(), normal.cpu().numpy(), inner_radius=0.5, outer_radius=1.0)
    mp = Map(rings=[ring])
    agent = Drone(init_state)

    if visualize:
        plt.ion()
        fig = plt.figure(figsize=(12, 6))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)
        plot_rings(ax3d, mp.rings)
        ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')

    history = []
    last_gate = -1

    while True:
        state_t = agent.get_state()
        history.append(state_t[:3].cpu().detach().numpy())

        if last_gate + 1 < len(mp.rings):
            center_t = mp.rings[last_gate + 1].center.to(DEVICE)
            if (state_t[:3] - center_t).norm().item() < pass_thresh:
                last_gate += 1
        done = (last_gate + 1 >= len(mp.rings))

        if done:
            if visualize:
                plt.close(fig)
            break

        sample = one_step(sample, center,model)
        state = sample_to_drone_state(sample, center)
        agent.state = state

        if visualize:
            ax2d.clear()
            s6 = agent.get_state().unsqueeze(0)
            yaw = s6[0, 5]
            euler = torch.tensor([0.0, 0.0, yaw], device=DEVICE, dtype=DTYPE)
            R_wc = euler_to_rot_matrix(*euler)
            R_cw = R_wc.T.unsqueeze(0)
            img_np = Environment(agent, mp).renderBatch(s6, R_cw)[0].cpu().detach().numpy()
            ax2d.imshow(img_np)
            ax2d.axis('off')

            ax3d.clear()
            plot_rings(ax3d, mp.rings)
            if history:
                hs = np.stack(history, axis=0)
                ax3d.plot(hs[:, 0], hs[:, 1], hs[:, 2], 'g-')
            draw_drone(ax3d, state[:3], torch.tensor([0.0, 0.0, state[5].item()], device=DEVICE))
            plt.pause(0.1)

    print("Test complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v_ref",             type=float, default=0.7)
    parser.add_argument("--dt",                type=float, default=0.1)
    parser.add_argument("--steps_per_episode", type=int,   default=200)
    parser.add_argument("--pass_thresh",       type=float, default=0.4)
    parser.add_argument("--visualize",         action="store_true")
    parser.add_argument("--no-visualize",      dest="visualize", action="store_false")
    parser.set_defaults(visualize=True)
    parser.add_argument("--rot_noise_std",     type=float, default=0.0)
    args = parser.parse_args()

    run_test(**vars(args))

