import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sampler import sample_interior_points, sample_boundary_points

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

N = 200
center = torch.tensor([0.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
normal = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
normal = normal / normal.norm()

samples, _, _ = sample_boundary_points(N, center, normal)
r   = samples[:, 0]
phi = samples[:, 1]
z   = samples[:, 2]
yaw = samples[:, 3]

x = r * torch.cos(phi+torch.pi)
y = r * torch.sin(phi+torch.pi)

head_dirs = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros(N, device=DEVICE)], dim=1)

xyz = torch.stack([x, y, z], dim=1).cpu().numpy()
dirs = head_dirs.cpu().numpy()

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=20, color='green')
ax.quiver(
    xyz[:, 0], xyz[:, 1], xyz[:, 2],
    dirs[:, 0], dirs[:, 1], dirs[:, 2],
    length=0.5, arrow_length_ratio=0.2, color='blue'
)
ax.set_box_aspect((1,1,1))

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
