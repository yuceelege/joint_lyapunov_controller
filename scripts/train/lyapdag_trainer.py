import argparse
import sys
from pathlib import Path
# Add project root to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add scripts/test to path to import propogate
sys.path.insert(0, str(Path(__file__).parent.parent / "test"))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sampler import sample_boundary_points, sample_interior_points
from propogate import one_step
from sublevel_sampler import sample_sublevel
from map import Map
from agent import Drone
from environment import Environment      # now the new Environment with renderBatch
from pid_controller import PIDController
from controller import SimpleControllerNN
from utils import *
from utils2 import *
import lyap
torch.autograd.set_detect_anomaly(True)

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

    # Lyapunov network and optimizer
    center = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    normal = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    eq = sample_sublevel(1, 0, center, normal)
    lyapu     = lyap.LyapunovNet(state_dim=7, xi_star=eq, epsilon=1e-3).to(device)
    lyap_opt  = optim.Adam(lyapu.parameters(), lr=1e-3)
    dagger_buf = []      # (obs, vel, rel, u_expert) pairs
    lyap_buf   = []      # adversarial xi ∈ ℝ⁸
    gamma      = 1.05        # sublevel expansion factor
    kappa      = 0.03        # exponential decrease rate
    c0, c1 = 0.1, 0.1    
    alpha      = 0.1         # PGD step size
    eta = 0.1
    K          = 5          # PGD iterations per seed

    time_indices = torch.arange(steps_per_episode, device=device)

    for ep in range(num_episodes):
        print(f"\n>>> Episode {ep+1}/{num_episodes}  β={beta:.3f}")

        # --- DAgger rollout to collect imitation data ---
        init_state = torch.zeros(7, device=device, dtype=dtype)
        init_pos_t = init_state[:3]

        rings      = generate_random_rings(init_state)
        mp         = Map(rings=rings)
        agent      = Drone(init_state)
        env        = Environment(agent, mp)
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
            print(step)
            state = agent.get_state()  # [6]
            # compute camera rotation from yaw
            yaw_t = state[5]                              # this is a 0‐D tensor
            body_euler = torch.stack([                   # still a single vector, but now grad‐connected
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(0.0, device=device, dtype=dtype),
                yaw_t
            ])
            R_wc  = euler_to_rot_matrix(*body_euler)      # now depends on yaw_t
            R_cw  = R_wc.T
            img_t = env.renderBatch(state.unsqueeze(0), R_cw.unsqueeze(0))[0]

            pos_t = state[:3]
            history.append(pos_t.cpu().numpy())

            epsilon = 0.1

            # if last_gate+1 < len(mp.rings) and \
            #    (pos_t - mp.rings[last_gate+1].center).norm() < pass_thresh:
            if last_gate+1 < len(mp.rings):
                ring   = mp.rings[last_gate+1]
                rel    = ring.center - pos_t
                axial  = torch.dot(ring.normal, rel)
                
                # exact projection: (I - nnᵀ)(p_i - x_t)
                I = torch.eye(3, device=rel.device, dtype=rel.dtype)
                P = I - ring.normal.unsqueeze(1) @ ring.normal.unsqueeze(0)
                proj = P @ rel
                radial = proj.norm()

                if 0.0 <= axial <= epsilon and radial <= ring.inner_radius:
                    # print("gate passed")
                    last_gate += 1


            idx      = idxs[min(ref_step, len(idxs)-1)]
            p_ref    = pts_t[idx]
            v_ref_pt = vel_ref_t[idx]
            done     = (last_gate+1 >= len(mp.rings))
            tang     = (pts_t[idx+1] - pts_t[idx]) if idx < len(pts_t)-1 else (pts_t[idx] - pts_t[idx-1])

            # orientation alignment
            bx      = R_wc[:,0] / R_wc[:,0].norm()
            tang_xy = tang.clone(); tang_xy[2] = 0
            tang_xy /= (tang_xy.norm() + 1e-8)
            ori_dot = torch.dot(bx, tang_xy)

            if done or (pos_t - p_ref).norm() > 0.6 or ori_dot < 0.8:
                rings      = generate_random_rings(init_state)
                mp         = Map(rings=rings)
                agent      = Drone(init_state)
                env        = Environment(agent, mp)
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

        # append new imitation data
        dataset.extend(episode_data)

        # --- Compute rho from boundary of B ---
        n_boundary = 20
        center = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)
        normal = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        boundary_xi, _, _ = sample_boundary_points(n_boundary, center, normal)
        adv_boundary = []
        for xi in boundary_xi:
            xi = xi.clone().detach().requires_grad_(True)  # xi on boundary
            for _ in range(K):
                V_xi = lyapu(xi.unsqueeze(0))                # [1, 1]
                grad = torch.autograd.grad(V_xi, xi)[0]      # ∇ξ V(ξ)
               # print(grad)
                xi = xi - eta * grad                        # gradient‐descent step on V
                with torch.no_grad():
                    xi.copy_(project_boundary(xi.unsqueeze(0), center, normal).squeeze(0))
                xi = xi.detach().requires_grad_(True)
            adv_boundary.append(xi.detach())

        adv_boundary = torch.stack(adv_boundary, dim=0)  # shape [n_boundary, 8]
        rho = compute_threshold(lyapu, adv_boundary, gamma)
        print(f"ρ = {rho.item():.6f}")

        n_interior = 100
        interior_xi, _, _ = sample_interior_points(n_interior, center, normal)
        interior_xi = interior_xi.to(device=device, dtype=dtype)
        for seed in interior_xi:
            xi = seed.clone().detach().requires_grad_(True)
            # xitest = xi.clone().detach().requires_grad_(True)
            # f_cltest = one_step(xitest, center,model)
            # losstest = f_cltest.sum()

            #     # 4. Backward to compute gradients w.r.t. xi
            # losstest.backward()

            #     # 5. Now xi.grad holds ∂(sum f_cl)/∂xi
            # print("∂(sum f_cl)/∂xi:\n", xitest.grad)
            for _ in range(K):
                f_cl = one_step(xi, center,model)

                loss_vio = violation_loss(
                    lyapu, f_cl.unsqueeze(0), xi.unsqueeze(0),
                    kappa, c0, rho, center, normal
                )
    
                grad = torch.autograd.grad(loss_vio, xi)[0]

                # print(grad)
                xi = xi + alpha * grad
                
                with torch.no_grad():
                    xi.copy_(project(xi.unsqueeze(0), center, normal).squeeze(0))
                
                xi = xi.detach().requires_grad_(True)
            # print("interior sampled")
            lyap_buf.append(xi.detach())

        cand = sample_sublevel(200, rho.item()*1.05, center, normal)
        cand = cand.to(device=device, dtype=dtype).detach()
        # compute ROA-growth loss
        VC     = lyapu(cand)
        L_roa  = torch.relu(VC / rho - 1).mean()
        
        
        print(f"Training on {len(dataset)} samples")
        for epoch in range(30):
            # sample imitation minibatch
            b      = min(len(dataset), 256)
            idxs_b = torch.randint(0, len(dataset), (b,), device=device)
            X_obs  = torch.stack([dataset[i][0] for i in idxs_b])
            X_vel  = torch.stack([dataset[i][1] for i in idxs_b])
            X_rel  = torch.stack([dataset[i][2] for i in idxs_b])
            Y      = torch.stack([dataset[i][3] for i in idxs_b])

            # sample lyap minibatch
            idxs_l = torch.randint(0, len(lyap_buf), (b,), device=device)
            XI     = torch.stack([lyap_buf[i] for i in idxs_l])

            # compute imitation loss
            pred   = model(X_obs, X_vel, X_rel)
            L_im   = loss_fn(pred, Y)

            # compute lyapunov-violation loss
            F_cl = torch.stack([one_step(xi, center, model) for xi in XI], dim=0)
            L_dotV = torch.stack([
                violation_loss(lyapu, F_cl[i].unsqueeze(0), XI[i].unsqueeze(0),
                               kappa, c0, rho, center, normal)
                for i in range(XI.size(0))
            ], dim=0).mean()

            cand = sample_sublevel(200, rho.item()*1.05, center, normal)
            cand = cand.to(device=device, dtype=dtype).detach()
            # compute ROA-growth loss
            VC     = lyapu(cand)
            L_roa  = torch.relu(VC / rho - 1).mean()
            
            # total loss and update
            total_loss = L_im + c0*L_dotV + c1 * L_roa
            print(
                f"L_im={L_im.item():.4f}, "
                f"L_dotV={c0*L_dotV.item():.4f}, "
                f"L_roa={c1*L_roa.item():.10f}, "
                f"total_loss={total_loss.item():.4f}")
            optimizer.zero_grad()
            lyap_opt.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()
            lyap_opt.step()
             
            print(f" Epoch {epoch+1}/30  total_loss={total_loss.item():.4f}")
        lyap_buf =[]  
        beta = max(beta * beta_decay, 0.0)
        if visualize:
            plt.close(fig)

    torch.save(model.state_dict(), '../../weights/joint_controller.pth')
    torch.save(lyapu.state_dict(),     '../../weights/joint_lyap.pth')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes",      type=int,   default=100)
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
