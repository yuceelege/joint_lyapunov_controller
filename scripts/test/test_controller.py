import argparse
import sys
from pathlib import Path
# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from map import Map, Ring
from agent import Drone
from environment import Environment
from controller import SimpleControllerNN
from utils import *
from propogate_visualize import drone_state_to_sample
from sublevel_sampler import *
from lyap import LyapunovNet
from sampler import *
from config import DEVICE, DTYPE

def run_test(v_ref=1.0,
             dt=0.1,
             sim_dt=0.1,
             steps_per_episode=200,
             pass_thresh=0.3,
             visualize=True,
             rot_noise_std=0.05):
    device = DEVICE
    last_pass_step = -1e9
    dwell_steps    = 10   # e.g. if T=2.0s and sim_dt=0.1s → dwell_steps=20
    model = SimpleControllerNN(color=True, image_size=50).to(device)
    model.load_state_dict(torch.load('../../weights/joint_controller.pth', map_location=device))
    model.eval()

    center = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    normal = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    eq = sample_sublevel(1, 0, center, normal)
    net     = LyapunovNet(state_dim=7, xi_star=eq, epsilon=1e-3).to(device)
    net.load_state_dict(torch.load('../../weights/joint_lyap.pth', map_location=device))
    net.eval()


    # init_state = torch.zeros(6, device=device, dtype=DTYPE)
    init_state = sample_init_state().to(device=device, dtype=DTYPE)
    init_pos_t = init_state[:3]

   # rings = generate_random_rings(init_state)
   # mp    = Map(rings=rings)
    mp = Map(ring_file="rings.json")
    agent = Drone(init_state)
    env   = Environment(agent, mp)

    traj_fn  = generate_trajectory(init_state[:3], mp)
    pts_t, _ = sample_trajectory(traj_fn, v_ref, dt, mp.rings)
    pts_np   = pts_t.cpu().numpy()
    vel_np   = np.gradient(pts_np, dt, axis=0)

    diffs     = np.linalg.norm(np.diff(pts_np, axis=0), axis=1)
    cum       = np.concatenate(([0.0], np.cumsum(diffs)))
    distances = np.clip(v_ref * sim_dt * np.arange(steps_per_episode), 0, cum[-1])
    idxs      = np.searchsorted(cum, distances)

    if visualize:
        plt.ion()
        # Main 3D and 2D figure
        fig   = plt.figure(figsize=(12,6))
        ax3d  = fig.add_subplot(1, 2, 1, projection='3d')
        ax3d.set_box_aspect((1,1,1))
        ax2d  = fig.add_subplot(1, 2, 2)

        plot_rings(ax3d, mp.rings)
        # Draw normal arrows for each ring
        for ring in mp.rings:
            center_np = ring.center.cpu().numpy()
            normal_np = ring.normal.cpu().numpy()
            ax3d.quiver(
                center_np[0], center_np[1], center_np[2],
                normal_np[0], normal_np[1], normal_np[2],
                length=1.0, arrow_length_ratio=0.2, color='r'
            )
        ax3d.plot(pts_np[:,0], pts_np[:,1], pts_np[:,2], 'b--', lw=2)

        # Transformed view + scalar plot figure
        fig2      = plt.figure(figsize=(10,6))
        axTrans   = fig2.add_subplot(1, 2, 1, projection='3d')
        axTrans.set_box_aspect((1,1,1))
        axScalar  = fig2.add_subplot(1, 2, 2)
        axScalar.set_xlabel('Step')
        axScalar.set_ylabel('V_custom')

    history    = []
    V_history  = []
    last_gate  = -1
    step       = ref_step = 0
    center_zero = torch.zeros(3, device=device, dtype=DTYPE)

    while True:
        state_t = agent.get_state()  # [6]
        # AFTER (yaw is differentiable)
        state_req = state_t.unsqueeze(0).clone().detach().requires_grad_(True)  # [1×6]
        yaw_req   = state_req[0, 5]                                              # tensor

        euler_req = torch.stack([torch.tensor(0.0, device=device),
                                torch.tensor(0.0, device=device),
                                yaw_req])
        R_wc_req  = euler_to_rot_matrix(*euler_req)
        R_cw_req  = R_wc_req.T

        img_grad  = env.renderBatch(state_req, R_cw_req.unsqueeze(0))[0]

        loss       = img_grad.sum()
        loss.backward()
        # print("∂loss/∂state:", state_req.grad)  # should be non-None
        img_t      = img_grad.detach()
        # --- GRAD-CHECK INSERTION END ---

        pos_t  = state_t[:3]

        history.append(pos_t.detach().cpu().numpy())

        # Analytical state-based gate passage check
        epsilon = 0.5
        print(step)
        # only try passing if enough steps have elapsed since last pass
        if last_gate+1 < len(mp.rings) and (step - last_pass_step) >= dwell_steps:
            ring  = mp.rings[last_gate+1]
            rel   = ring.center - pos_t  # relative vector from ring center to drone position
            axial = torch.dot(ring.normal, rel)

            I     = torch.eye(3, device=rel.device, dtype=rel.dtype)
            P     = I - ring.normal.unsqueeze(1) @ ring.normal.unsqueeze(0)
            proj  = P @ rel
            radial= proj.norm()

            if 0.0 <= axial <= epsilon and radial <= ring.inner_radius:
                print("gate passed")
                last_gate      += 1
                last_pass_step  = step


        done   = (last_gate + 1 >= len(mp.rings))

        if done:
            if visualize:
                plt.close(fig)
                plt.close(fig2)
            # reset
            # rings = generate_random_rings(init_state)
            # mp    = Map(rings=rings)
            # init_state = torch.zeros(6, device=device, dtype=DTYPE)
            init_state = sample_init_state().to(device=device, dtype=DTYPE)
            mp = Map(ring_file="rings.json")
            agent = Drone(init_state)
            env   = Environment(agent, mp)
            history = []
            V_history = []
            last_gate = -1
            ref_step = 0

            traj_fn  = generate_trajectory(init_state[:3], mp)
            pts_t, _ = sample_trajectory(traj_fn, v_ref, dt, mp.rings)
            pts_np   = pts_t.cpu().numpy()
            vel_np   = np.gradient(pts_np, dt, axis=0)
            diffs     = np.linalg.norm(np.diff(pts_np, axis=0), axis=1)
            cum       = np.concatenate(([0.0], np.cumsum(diffs)))
            distances = np.clip(v_ref * sim_dt * np.arange(steps_per_episode), 0, cum[-1])
            idxs      = np.searchsorted(cum, distances)

            if visualize:
                plt.ion()
                fig   = plt.figure(figsize=(12,6))
                ax3d  = fig.add_subplot(1, 2, 1, projection='3d')
                ax3d.set_box_aspect((1,1,1))
                ax2d  = fig.add_subplot(1, 2, 2)
                plot_rings(ax3d, mp.rings)
                for ring in mp.rings:
                    center_np = ring.center.cpu().numpy()
                    normal_np = ring.normal.cpu().numpy()
                    ax3d.quiver(
                        center_np[0], center_np[1], center_np[2],
                        normal_np[0], normal_np[1], normal_np[2],
                        length=1.0, arrow_length_ratio=0.2, color='r'
                    )
                ax3d.plot(pts_np[:,0], pts_np[:,1], pts_np[:,2], 'b--', lw=2)

                fig2      = plt.figure(figsize=(10,6))
                axTrans   = fig2.add_subplot(1, 2, 1, projection='3d')
                axTrans.set_box_aspect((1,1,1))
                axScalar  = fig2.add_subplot(1, 2, 2)
                axScalar.set_xlabel('Step')
                axScalar.set_ylabel('V_custom')

            step += 1
            continue

        img = img_t.cpu().numpy()

        # rest of your loop unchanged...
        obs_t    = torch.from_numpy(img.transpose(2,0,1)).to(device)
        r, vz, th, yaw_rate = state_t[3], state_t[4], state_t[5], state_t[6]

        lin_vel_t = torch.stack([
            r * torch.cos(th),
            r * torch.sin(th),
            vz,
            yaw_rate
        ], dim=0).to(device) 

        prev_t = mp.rings[last_gate].center if last_gate>=0 else init_pos_t
        next_t = mp.rings[last_gate+1].center if last_gate+1<len(mp.rings) else pos_t
        rel_t  = (next_t - prev_t).to(device)

        u_policy = model(obs_t.unsqueeze(0),
                              lin_vel_t.unsqueeze(0),
                              rel_t.unsqueeze(0)).squeeze(0)

        agent.update_state(u_policy, sim_dt)
        noisy = agent.get_state().clone()
        noisy[:3] += torch.normal(0.0, rot_noise_std, (3,), device=device)
        noisy[5]  += torch.normal(0.0, rot_noise_std, (),   device=device)
        agent.state = noisy

        if visualize:
            # Update main views
            ax2d.clear();     ax2d.imshow(img); ax2d.axis('off')
            ax3d.clear()
            plot_rings(ax3d, mp.rings)
            for ring in mp.rings:
                center_np = ring.center.cpu().numpy()
                normal_np = ring.normal.cpu().numpy()
                ax3d.quiver(
                    center_np[0], center_np[1], center_np[2],
                    normal_np[0], normal_np[1], normal_np[2],
                    length=1.0, arrow_length_ratio=0.2, color='r'
                )
            draw_drone(ax3d, pos_t,
                       torch.tensor([0.0,0.0,state_t[5].item()], device=device))
            if history:
                h = np.stack(history, axis=0)
                ax3d.plot(h[:,0], h[:,1], h[:,2], 'g-')

            # Compute transformation based on next gate center & normal
            if last_gate+1 < len(mp.rings):
                center_next = mp.rings[last_gate+1].center.cpu().numpy()
                normal_next = mp.rings[last_gate+1].normal.cpu().numpy()
            else:
                center_next = np.zeros(3)
                normal_next = np.array([1.0, 0.0, 0.0])

            # Translate drone position so that center_next -> origin
            pos_np = pos_t.detach().cpu().numpy()
            rel_pos = pos_np - center_next

            # Compute rotation matrix that sends normal_next to [1,0,0]
            tgt = np.array([1.0, 0.0, 0.0])
            n = normal_next / np.linalg.norm(normal_next)
            dot = np.clip(np.dot(n, tgt), -1.0, 1.0)
            if np.allclose(n, tgt):
                R = np.eye(3)
            elif np.allclose(n, -tgt):
                R = np.array([[-1,  0,  0],
                              [ 0, -1,  0],
                              [ 0,  0,  1]])
            else:
                axis = np.cross(n, tgt)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(dot)
                K = np.array([[    0,    -axis[2],  axis[1]],
                              [ axis[2],     0,    -axis[0]],
                              [-axis[1],  axis[0],     0   ]])
                R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)

            # Apply rotation to the translated position
            pos_transformed = R @ rel_pos

            # Compute and transform the drone's heading direction
            yaw = state_t[5].item()
            heading_world = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            heading_transformed = R @ heading_world

            # Compute new yaw from transformed heading
            new_yaw = np.arctan2(heading_transformed[1], heading_transformed[0])

            yaw0    = state_t[5]
            dr0     = state_t[3]
            dz0     = state_t[4]

            v_world = torch.stack([
                dr0 * torch.cos(yaw0),
                dr0 * torch.sin(yaw0),
                dz0
            ], dim=0)

            R_torch = torch.from_numpy(R).to(device=device, dtype=DTYPE)
            v_gate   = R_torch @ v_world

            dr       = v_gate[:2].norm()
            dz       = v_gate[2]
            dyaw     = state_t[6]


            state_transformed = torch.stack([
                torch.tensor(pos_transformed[0], device=device, dtype=DTYPE),
                torch.tensor(pos_transformed[1], device=device, dtype=DTYPE),
                torch.tensor(pos_transformed[2], device=device, dtype=DTYPE),
                dr,
                dz,
                torch.tensor(new_yaw,      device=device, dtype=DTYPE),
                dyaw
            ], dim=0)
                                            
            
            xi_torch = drone_state_to_sample(state_transformed, center_zero)   # shape [7]
            with torch.no_grad():
                xi_t = xi_torch.unsqueeze(0).to(device)                        # now [1,7]
                V_val = net(xi_t).cpu().numpy()[0]                             # scalar from batch


            V_history.append(V_val)

            # Update transformed view
            axTrans.clear()
            # Plot the transformed drone as a point
            axTrans.scatter(
                pos_transformed[0],
                pos_transformed[1],
                pos_transformed[2],
                color='m', s=30
            )
            # Plot the transformed normal (X-axis) as an arrow from origin
            axTrans.quiver(
                0, 0, 0,
                1, 0, 0,
                length=1.0, arrow_length_ratio=0.2, color='r'
            )
            # Plot the transformed heading arrow from the drone
            axTrans.quiver(
                pos_transformed[0], pos_transformed[1], pos_transformed[2],
                heading_transformed[0], heading_transformed[1], heading_transformed[2],
                length=0.5, arrow_length_ratio=0.2, color='b'
            )
            axTrans.set_xlim(-2, 2)
            axTrans.set_ylim(-2, 2)
            axTrans.set_zlim(-2, 2)

            # Update scalar plot
            axScalar.clear()
            axScalar.plot(V_history, 'm-')
            axScalar.set_xlabel('Step')
            axScalar.set_ylabel('V_custom')

            plt.pause(0.001)

        step += 1
        ref_step += 1

    print("Test complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v_ref",             type=float, default=0.7)
    parser.add_argument("--dt",                type=float, default=0.1)
    parser.add_argument("--sim_dt",            type=float, default=0.1)
    parser.add_argument("--steps_per_episode", type=int,   default=200)
    parser.add_argument("--pass_thresh",       type=float, default=0.4)
    parser.add_argument("--visualize",         action="store_true")
    parser.add_argument("--no-visualize",      dest="visualize", action="store_false")
    parser.set_defaults(visualize=True)
    parser.add_argument("--rot_noise_std",     type=float, default=0.0)
    args = parser.parse_args()

    run_test(**vars(args))
