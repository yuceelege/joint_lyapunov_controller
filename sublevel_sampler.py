import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sampler import sample_interior_points


# shared weights per variable
WR, WPHI, WZ, WYAW, WDYAW, WDZ, WDR = 0.25, 0.1, 0.25, 0.15, 0.08, 0.08, 0.025

# Custom potential

def V_custom(xi):
    r, phi, z, yaw, dr, dz, dyaw = xi.unbind(-1)
    return (
        WR    * r**2
      + WPHI  * phi**2
      + WZ    * z**2
      + WYAW  * yaw**2
      + WDYAW * dyaw**2
      + WDZ   * dz**2
      + WDR   * (dr - 0.5)**2
    )

# Sublevel sampler aligned with V_custom

def sample_sublevel(N, c, center, normal):
    samples = np.zeros((N, 7), dtype=np.float32)
    for i in range(N):
        u = np.random.exponential(size=5).astype(np.float32)
        u_sum = u.sum()
        c_r, c_phi, c_z, c_yaw, c_v = c * (u / u_sum)

        r   = np.sqrt(c_r / WR)
        phi = np.random.randn() * np.sqrt(c_phi / WPHI)
        z   = np.random.randn() * np.sqrt(c_z / WZ)
        yaw = np.random.randn() * np.sqrt(c_yaw / WYAW)

        v_mag = np.sqrt(c_v / WDR)
        g3 = np.random.randn(3).astype(np.float32)
        g3 /= np.linalg.norm(g3)
        dr   = 0.5 + g3[0] * v_mag
        dz   = g3[1] * v_mag
        dyaw = g3[2] * v_mag

        samples[i] = [r, phi, z, yaw, dr, dz, dyaw]

    return torch.from_numpy(samples).to(center.device)

# Demonstration plot

if __name__ == "__main__":
    center = torch.zeros(3, dtype=torch.float32)
    normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    xi      = sample_sublevel(1, 0, center, normal)
    r_vals  = xi[:, 0].cpu().numpy()
    phi     = xi[:, 1].cpu().numpy()
    z_vals  = xi[:, 2].cpu().numpy()
    yaw_vals= xi[:, 3].cpu().numpy()

    # Compute Cartesian for plotting (offset phi by π)
    phi_plot = phi + np.pi
    x = r_vals * np.cos(phi_plot) + center[0].item()
    y = r_vals * np.sin(phi_plot) + center[1].item()
    pts3 = np.stack([x, y, z_vals + center[2].item()], axis=1)

    vals    = V_custom(xi)
    vals_np = vals.cpu().numpy()

    dirs = np.stack([np.cos(yaw_vals), np.sin(yaw_vals), np.zeros_like(yaw_vals)], axis=1)

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(
        pts3[:,0], pts3[:,1], pts3[:,2],
        c=vals_np, cmap='viridis', s=20
    )
    ax.quiver(
        pts3[:,0], pts3[:,1], pts3[:,2],
        dirs[:,0], dirs[:,1], dirs[:,2],
        length=0.5, arrow_length_ratio=0.2, color='black'
    )
    plt.colorbar(sc, ax=ax, shrink=0.5)
    ax.set_box_aspect([1,1,1])
    plt.show()
