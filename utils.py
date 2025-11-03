import torch
import math
import numpy as np
from map import Ring, Map
from typing import Tuple, List
from scipy.interpolate import splprep, splev

# Device and dtype configuration (imported from config)
from config import device, dtype, DEVICE, DTYPE

def euler_to_rot_matrix(roll: torch.Tensor,
                        pitch: torch.Tensor,
                        yaw: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to a 3×3 rotation matrix (all in PyTorch)."""
    roll, pitch, yaw = roll.to(device, dtype), pitch.to(device, dtype), yaw.to(device, dtype)
    cphi, sphi = torch.cos(roll), torch.sin(roll)
    cth,  sth  = torch.cos(pitch), torch.sin(pitch)
    cy,   sy   = torch.cos(yaw),   torch.sin(yaw)

    row0 = torch.stack([ cy*cth,
                         cy*sth*sphi - sy*cphi,
                         cy*sth*cphi + sy*sphi ], dim=0)
    row1 = torch.stack([ sy*cth,
                         sy*sth*sphi + cy*cphi,
                         sy*sth*cphi - cy*sphi ], dim=0)
    row2 = torch.stack([ -sth,
                          cth*sphi,
                          cth*cphi ], dim=0)

    return torch.stack([row0, row1, row2], dim=0).to(device, dtype)

def draw_drone(ax, pos: torch.Tensor, euler: torch.Tensor, scale: float = 0.5):
    pos = pos.to(device, dtype)
    euler = euler.to(device, dtype)
    R = euler_to_rot_matrix(euler[0], euler[1], euler[2])
    axes = [R[:, i] * scale for i in range(3)]
    cols = ['r', 'g', 'b']
    quivers = []
    pos_np = pos.detach().cpu().numpy()
    for v, c in zip(axes, cols):
        v_np = v.detach().cpu().numpy()
        quivers.append(
            ax.quiver(
                pos_np[0], pos_np[1], pos_np[2],
                v_np[0], v_np[1], v_np[2],
                color=c, linewidth=2
            )
        )
    return quivers

def compute_desired_orientation(tangent: torch.Tensor) -> torch.Tensor:
    t = tangent.to(device, dtype).clone()
    t[2] = 0.0
    norm = torch.norm(t)
    if norm.item() < 1e-6:
        yaw = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        yaw = torch.atan2(t[1], t[0])
    return euler_to_rot_matrix(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(0.0, device=device, dtype=dtype),
        yaw
    )

def rotation_matrix_to_euler_angles(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R = R.to(device, dtype)
    yaw   = torch.atan2(R[1,0], R[0,0])
    pitch = torch.atan2(-R[2,0], torch.sqrt(R[2,1]**2 + R[2,2]**2))
    roll  = torch.atan2(R[2,1], R[2,2])
    return roll, pitch, yaw

def hat(v: torch.Tensor) -> torch.Tensor:
    v = v.to(device, dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    return torch.tensor([
        [ zero,    -v[2],  v[1]],
        [  v[2],    zero, -v[0]],
        [ -v[1],    v[0],  zero]
    ], device=device, dtype=dtype)

def vee(M: torch.Tensor) -> torch.Tensor:
    M = M.to(device, dtype)
    return torch.stack([M[2,1], M[0,2], M[1,0]], dim=0)

def generate_trajectory(initial_pos: torch.Tensor, map_instance: Map):
    init_np = initial_pos.to(device, dtype).cpu().numpy()
    knots = [init_np] + [r.center.cpu().numpy() for r in map_instance.rings]
    pts = np.vstack(knots).T
    dists = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=1), axis=0))))
    u = dists / dists[-1]
    tck, _ = splprep(pts, u=u, k=2, s=0)

    def traj(t: torch.Tensor) -> torch.Tensor:
        tt = t.to(device, dtype).cpu().numpy()
        Pn = np.array(splev(tt, tck)).T
        P = torch.from_numpy(Pn).to(device, dtype)
        return P[0] if P.ndim == 1 or P.shape[0] == 1 else P

    return traj
def generate_trajectory2(initial_pos: torch.Tensor, map_instance: Map):
    init_np = initial_pos.to(device, dtype).cpu().numpy()
    knots = [init_np] + [r.center.cpu().numpy() for r in map_instance.rings]
    pts = np.vstack(knots).T
    dists = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=1), axis=0))))
    u = dists / dists[-1]
    tck, _ = splprep(pts, u=u, k=1, s=0)

    def traj(t: torch.Tensor) -> torch.Tensor:
        tt = t.to(device, dtype).cpu().numpy()
        Pn = np.array(splev(tt, tck)).T
        P = torch.from_numpy(Pn).to(device, dtype)
        return P[0] if P.ndim == 1 or P.shape[0] == 1 else P

    return traj

def sample_trajectory(spline, v_ref: float, dt: float, rings: List[Ring]):
    n = 5000
    t_np = np.linspace(0.0, 1.0, n)
    t_tensor = torch.from_numpy(t_np).to(device, dtype)
    pts = spline(t_tensor)
    pts_np = pts.cpu().numpy()
    diffs = np.diff(pts_np, axis=0)
    L = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(L)))
    total = cum[-1]
    steps = np.arange(0.0, total, v_ref * dt)
    if steps[-1] < total:
        steps = np.append(steps, total)
    ts = np.interp(steps, cum, t_np)
    sampled_t = spline(torch.from_numpy(ts).to(device, dtype))
    sampled_np = sampled_t.cpu().numpy()
    idx = [int(np.argmin([np.linalg.norm(p - r.center.cpu().numpy()) for r in rings])) for p in sampled_np]
    idx_t = torch.from_numpy(np.array(idx, dtype=int)).to(torch.long)
    return sampled_t, idx_t

def generate_trajectory_bezier(initial_pos: torch.Tensor, map_instance: Map):
    init_np = initial_pos.to(device, dtype).cpu().numpy()
    knots = [init_np] + [r.center.cpu().numpy() for r in map_instance.rings]
    normals = [r.normal.cpu().numpy() for r in map_instance.rings]
    if not normals:
        normals = [np.array([1.0, 0.0, 0.0])]
    tangents = [normals[0]] + normals
    chord = np.linalg.norm(np.diff(knots, axis=0), axis=1)
    u = np.concatenate(([0.0], np.cumsum(chord))) / chord.sum()
    segs = []
    for i in range(len(knots)-1):
        B0, B3 = knots[i], knots[i+1]
        h = chord[i] / 3.0
        B1 = B0 + h * tangents[i]
        B2 = B3 - h * tangents[i+1]
        segs.append((B0, B1, B2, B3, u[i], u[i+1]))

    def traj(t: torch.Tensor) -> torch.Tensor:
        tt = t.to(device, dtype).cpu().numpy()
        P = np.zeros((len(tt), 3))
        for j, tv in enumerate(tt):
            for B0, B1, B2, B3, t0, t1 in segs:
                if t0 <= tv <= t1:
                    s = (tv - t0) / (t1 - t0)
                    P[j] = (1-s)**3 * B0 + 3*(1-s)**2 * s * B1 \
                         + 3*(1-s) * s**2 * B2 + s**3 * B3
                    break
            else:
                P[j] = segs[0][0] if tv < tt[0] else segs[-1][3]
        Pt = torch.from_numpy(P).to(device, dtype)
        return Pt[0] if Pt.ndim == 1 or Pt.shape[0] == 1 else Pt

    return traj

def plot_rings(ax, rings: List[Ring]):
    for ring in rings:
        n = torch.tensor(ring.normal, device=device, dtype=dtype)
        arb = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        if torch.abs(torch.dot(n, arb)) > 0.9:
            arb = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        u = torch.cross(n, arb)
        u = u / torch.norm(u)
        v = torch.cross(n, u)
        theta = torch.linspace(0.0, 2*math.pi, 100, device=device, dtype=dtype)
        radii = torch.linspace(ring.inner_radius, ring.outer_radius, 10,
                               device=device, dtype=dtype)
        Rg, Tg = torch.meshgrid(radii, theta, indexing='ij')
        cosT, sinT = torch.cos(Tg), torch.sin(Tg)
        X = ring.center[0] + cosT * Rg * u[0] + sinT * Rg * v[0]
        Y = ring.center[1] + cosT * Rg * u[1] + sinT * Rg * v[1]
        Z = ring.center[2] + cosT * Rg * u[2] + sinT * Tg * v[2]
        ax.plot_surface(
            X.cpu().numpy(),
            Y.cpu().numpy(),
            Z.cpu().numpy(),
            color='red',
            alpha=0.5,
            edgecolor='none'
        )

def generate_random_rings(initial_state):
    ip = torch.tensor(initial_state[:3], device=device, dtype=dtype)
    rings = []
    prev_center = ip.clone()
    prev_normal = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    count = torch.randint(2, 4, (1,)).item()
    z_var = 0.3
    thr_n, thr_d = 0.05, 0.05
    for _ in range(count):
        while True:
            phi = torch.rand(1).item() * 2 * math.pi
            normal = torch.tensor([math.cos(phi), math.sin(phi), 0.0],
                                   device=device, dtype=dtype)
            if (prev_normal @ normal) < 1 - thr_n:
                continue
            psi = torch.rand(1).item() * 2 * math.pi
            r = torch.rand(1).item() * 0.5 + 4.0
            disp_xy = torch.tensor([math.cos(psi), math.sin(psi), 0.0],
                                   device=device, dtype=dtype) * r
            disp_dir = disp_xy / torch.norm(disp_xy)
            if (prev_normal @ disp_dir) <= 1 - thr_d:
                continue
            dz = (torch.rand(1).item() * 2 - 1) * z_var
            disp = disp_xy + torch.tensor([0.0, 0.0, dz],
                                          device=device, dtype=dtype)
            break
        center = prev_center + disp
        rings.append(Ring(center.cpu().numpy(), normal.cpu().numpy(), inner_radius=0.5, outer_radius=1))
        prev_center = center
        prev_normal = normal
    return rings

def generate_random_rings2(initial_state):
    ip = torch.tensor(initial_state[:3], device=device, dtype=dtype)
    rings = []
    prev_center = ip.clone()
    prev_normal = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    count = torch.randint(1, 2, (1,)).item()
    z_var = 0.3
    thr_n, thr_d = 0.05, 0.05
    for _ in range(count):
        while True:
            phi = torch.rand(1).item() * 2 * math.pi
            normal = torch.tensor([math.cos(phi), math.sin(phi), 0.0],
                                   device=device, dtype=dtype)
            if (prev_normal @ normal) < 1 - thr_n:
                continue
            psi = torch.rand(1).item() * 2 * math.pi
            r = torch.rand(1).item() * 0.5 + 4.0
            disp_xy = torch.tensor([math.cos(psi), math.sin(psi), 0.0],
                                   device=device, dtype=dtype) * r
            disp_dir = disp_xy / torch.norm(disp_xy)
            if (prev_normal @ disp_dir) <= 1 - thr_d:
                continue
            dz = (torch.rand(1).item() * 2 - 1) * z_var
            disp = disp_xy + torch.tensor([0.0, 0.0, dz],
                                          device=device, dtype=dtype)
            break
        center = prev_center + disp
        rings.append(Ring(center.cpu().numpy(), normal.cpu().numpy(), inner_radius=0.5, outer_radius=1))
        prev_center = center
        prev_normal = normal
    return rings

def state8_to_state12(s8) -> torch.Tensor:
    """
    Convert an 8-D XYDrone state into a 12-D [pos, vel, euler, omega] vector.
    """
    s8 = torch.as_tensor(s8, device=device, dtype=dtype)
    s12 = torch.zeros(12, device=device, dtype=dtype)
    s12[0] = s8[4]
    s12[1] = s8[5]
    s12[2] = s8[0]
    s12[3] = s8[6]
    s12[4] = s8[7]
    s12[5] = s8[1]
    s12[6] = 0.0
    s12[7] = 0.0
    s12[8] = s8[2]
    s12[9]  = 0.0
    s12[10] = 0.0
    s12[11] = s8[3]
    return s12

def sample_init_state():
    x = (2*torch.rand(1) - 1)*0.5
    y = (2*torch.rand(1) - 1)*0.5
    z = (2*torch.rand(1) - 1)*0.1
    vr = torch.rand(1)*0.25 +0.25
    vz = torch.rand(1)*0.25
    yaw = -math.pi/6 + torch.rand(1)*(math.pi/3)
    dyaw = (torch.rand(1)-1)*0.1
    return torch.cat([x, y, z, vr, vz, yaw, dyaw])