import argparse
import sys
from pathlib import Path
# Add project root to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from map import Map
from agent import Drone
from environment import Environment      # now the new Environment with renderBatch
from pid_controller import PIDController
from controller import SimpleControllerNN
from utils import (
    generate_random_rings,
    generate_trajectory,
    sample_trajectory,
    draw_drone,
    plot_rings,
    device,
    dtype,
    euler_to_rot_matrix
)


def run_dagger(num_episodes: int = 5,
               v_ref: float = 1.0,
               dt: float = 0.1,
               sim_dt: float = 0.1,
               steps_per_episode: int = 200,
               pass_thresh: float = 0.25,
               beta_start: float = 0.95,
               beta_decay: float = 0.99,
               visualize: bool = True,
               rot_noise_std: float = 0.05):

    model     = SimpleControllerNN(color=True, image_size=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()
    dataset   = []
    beta      = beta_start

    time_indices = torch.arange(steps_per_episode, device=device)

    for ep in range(num_episodes):
        print(f"\n>>> Episode {ep+1}/{num_episodes}  β={beta:.3f}")

        init_state = torch.zeros(7, device=device, dtype=dtype)
        init_pos_t = init_state[:3]

        # rings      = generate_random_rings(init_state)
        # mp         = Map(rings=rings)
        mp = Map(ring_file="rings.json")
        agent      = Drone(init_state)
        env        = Environment(agent,mp)              # only map passed now
        controller = PIDController(dt=sim_dt)

        traj_fn, _ = generate_trajectory(init_state[:3], mp), None
        pts_t, _   = sample_trajectory(traj_fn, v_ref, dt, mp.rings)
        pts_np     = pts_t.cpu().numpy()

        diffs_pts = torch.diff(pts_t, dim=0) / dt
        vel_ref_t = torch.cat([diffs_pts, diffs_pts[-1:].clone()], dim=0)

        arc_dists = torch.norm(torch.diff(pts_t, dim=0), dim=1)
        cum       = torch.cat([torch.tensor([0.0], device=device, dtype=dtype),
                               torch.cumsum(arc_dists, dim=0)])
        distances = (v_ref * sim_dt * time_indices).clamp(max=cum[-1])
        idxs      = torch.searchsorted(cum, distances)

        if visualize:
            plt.ion()
            fig  = plt.figure(figsize=(12,6))
            ax3d = fig.add_subplot(1,2,1, projection='3d')
            ax2d = fig.add_subplot(1,2,2)
            plot_rings(ax3d, mp.rings)
            t_vis   = torch.linspace(0, 1, steps_per_episode, device=device)
            pts_vis = traj_fn(t_vis).cpu().numpy()
            ax3d.plot(pts_vis[:,0], pts_vis[:,1], pts_vis[:,2], 'b--', lw=2)

        history, last_gate = [], -1
        step, ref_step    = 0, 0
        episode_data      = []

        while step < steps_per_episode:
            print(f"Step: {step}")

            state = agent.get_state()                               # [6]
            # compute camera rotation from yaw
            body_euler = torch.tensor([0.0, 0.0, state[5]], device=device, dtype=dtype)
            R_wc       = euler_to_rot_matrix(*body_euler)
            R_cw       = R_wc.T

            # renderBatch returns [1,H,W,3]; take the first element
            img_t = env.renderBatch(state.unsqueeze(0), R_cw.unsqueeze(0))[0]

            pos_t  = state[:3]
            history.append(pos_t.cpu().numpy())

            if last_gate+1 < len(mp.rings) and \
               (pos_t - mp.rings[last_gate+1].center).norm() < pass_thresh:
                last_gate += 1

            idx      = idxs[min(ref_step, len(idxs)-1)]
            p_ref    = pts_t[idx]
            v_ref_pt = vel_ref_t[idx]
            done     = (last_gate+1 >= len(mp.rings))
            tang     = (pts_t[idx+1] - pts_t[idx]) if idx < len(pts_t)-1 else (pts_t[idx] - pts_t[idx-1])

            # orientation alignment
            bx      = R_wc[:,0] / R_wc[:,0].norm()
            tang_xy = tang.clone(); tang_xy[2]=0
            tang_xy /= (tang_xy.norm() + 1e-8)
            ori_dot = torch.dot(bx, tang_xy)

            if done or (pos_t - p_ref).norm() > 0.6 or ori_dot < 0.8:
                # rings      = generate_random_rings(init_state)
                # mp         = Map(rings=rings)
                mp = Map(ring_file="rings.json")
                agent      = Drone(init_state)
                env        = Environment(agent,mp)
                controller = PIDController(dt=sim_dt)
                history, last_gate = [], -1
                ref_step  = 0

                traj_fn, _ = generate_trajectory(init_state[:3], mp), None
                pts_t, _   = sample_trajectory(traj_fn, v_ref, dt, mp.rings)
                pts_np     = pts_t.cpu().numpy()
                diffs_pts  = torch.diff(pts_t, dim=0) / dt
                vel_ref_t  = torch.cat([diffs_pts, diffs_pts[-1:].clone()], dim=0)
                arc_dists  = torch.norm(torch.diff(pts_t, dim=0), dim=1)
                cum        = torch.cat([torch.tensor([0.0], device=device, dtype=dtype),
                                        torch.cumsum(arc_dists, dim=0)])
                distances  = (v_ref * sim_dt * time_indices).clamp(max=cum[-1])
                idxs       = torch.searchsorted(cum, distances)
                step += 1
                continue

            u_expert = controller.compute_control(state, p_ref, v_ref_pt, tang, dref=None)

            obs_t = img_t.permute(2,0,1).unsqueeze(0)
            r, vz, th, yaw_rate = state[3], state[4], state[5], state[6]

            lin_vel = torch.stack([
                r * torch.cos(th),
                r * torch.sin(th),
                vz,
                yaw_rate
            ], dim=0)            # → a 4-vector now

            vel_in  = lin_vel.unsqueeze(0)
            prev_c    = mp.rings[last_gate].center if last_gate>=0 else init_pos_t
            next_c    = mp.rings[last_gate+1].center
            rel       = (next_c - prev_c).unsqueeze(0)

            with torch.no_grad():
                u_policy = model(obs_t, vel_in, rel).squeeze(0)

            u = beta*u_expert + (1-beta)*u_policy
            episode_data.append((obs_t.squeeze(0),
                                 vel_in.squeeze(0),
                                 rel.squeeze(0),
                                 u_expert))

            agent.update_state(u, sim_dt)
            noisy = agent.get_state().clone()
            noisy[:3] += torch.normal(0.0, rot_noise_std, (3,), device=device, dtype=dtype)
            noisy[5]  += torch.normal(0.0, rot_noise_std, (),   device=device, dtype=dtype)
            agent.state = noisy

            if visualize:
                ax2d.clear(); ax2d.imshow(img_t.cpu().numpy()); ax2d.axis('off')
                ax3d.clear(); plot_rings(ax3d, mp.rings)
                ax3d.plot(pts_np[:,0], pts_np[:,1], pts_np[:,2], 'b--', lw=2)
                draw_drone(ax3d,
                           agent.get_state()[:3],
                           torch.tensor([0,0,agent.get_state()[5]], device=device, dtype=dtype))
                if history:
                    h = torch.stack([torch.tensor(h) for h in history]).cpu().numpy()
                    ax3d.plot(h[:,0],h[:,1],h[:,2], 'g-')
                plt.pause(1e-3)

            step += 1
            ref_step += 1

        dataset.extend(episode_data)
        N = len(dataset)
        print(f"Training on {N} samples")
        for epoch in range(30):
            b      = min(N, 256)
            idxs_b = torch.randint(0, N, (b,), device=device)
            X_obs  = torch.stack([dataset[i][0] for i in idxs_b]).to(device)
            X_vel  = torch.stack([dataset[i][1] for i in idxs_b]).to(device)
            X_rel  = torch.stack([dataset[i][2] for i in idxs_b]).to(device)
            Y      = torch.stack([dataset[i][3] for i in idxs_b]).to(device)
            pred   = model(X_obs, X_vel, X_rel)
            loss   = loss_fn(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f" Epoch {epoch+1}/30  loss={loss.item():.4f}")

        beta = max(beta * beta_decay, 0.0)
        if visualize:
            plt.close(fig)

    torch.save(model.state_dict(), '../../weights/dagger_weights.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes",      type=int,   default=95)
    parser.add_argument("--v_ref",             type=float, default=0.7)
    parser.add_argument("--dt",                type=float, default=0.1)
    parser.add_argument("--sim_dt",            type=float, default=0.1)
    parser.add_argument("--steps_per_episode", type=int,   default=200)
    parser.add_argument("--pass_thresh",       type=float, default=0.4)
    parser.add_argument("--beta_start",        type=float, default=1)
    parser.add_argument("--beta_decay",        type=float, default=0.98)
    parser.add_argument("--visualize",         action="store_true")
    parser.add_argument("--no-visualize",      dest="visualize", action="store_false")
    parser.set_defaults(visualize=False)
    parser.add_argument("--rot_noise_std",     type=float, default=0.01)
    args = parser.parse_args()

    run_dagger(**vars(args))
