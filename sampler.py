import torch
epsilon = 0.3
r_min, r_max = epsilon, 4.0
phi_min, phi_max = -1/4 * torch.pi, 1/4 * torch.pi
z_min, z_max = -0.5, 0.5
delta_yaw = torch.deg2rad(torch.tensor(30.0))


dr_max = 1.0 
dz_max = 0.8
dyaw_max = 1.2

def sample_interior_points(N, center, normal):
    device, dtype = center.device, center.dtype
    center, normal = -center, -normal
    r = torch.rand(N, device=device, dtype=dtype) * (r_max - r_min) + r_min
    phi = torch.rand(N, device=device, dtype=dtype) * (phi_max - phi_min) + phi_min
    z = torch.rand(N, device=device, dtype=dtype) * (z_max - z_min) + z_min
    yaw_min = phi - delta_yaw
    yaw_max = phi + delta_yaw
    yaw = torch.rand(N, device=device, dtype=dtype) * (yaw_max - yaw_min) + yaw_min
    dr = torch.rand(N, device=device, dtype=dtype)*1 + 0.3
    dz = (torch.rand(N, device=device, dtype=dtype) - 0.5) * dz_max
    dyaw = (torch.rand(N, device=device, dtype=dtype) - 0.5) * dyaw_max
    out = torch.stack([r, phi, z, yaw, dr, dz, dyaw], dim=1)
    return out, center, normal


def sample_candidate_S(N, center, normal):
    dz_max = 0.6
    phi_min, phi_max = -1/4.5 * torch.pi, 1/4.5 * torch.pi
    device, dtype = center.device, center.dtype
    center, normal = -center, -normal
    r = torch.rand(N, device=device, dtype=dtype) * (r_max - r_min) + r_min
    r_avg = r/r_max
    delta_yaw1 = delta_yaw * r_avg
    phi = torch.rand(N, device=device, dtype=dtype) * (phi_max - phi_min) + phi_min
    z = torch.rand(N, device=device, dtype=dtype) * (z_max - z_min) + z_min
    yaw_min = phi - delta_yaw1
    yaw_max = phi + delta_yaw1
    yaw = torch.rand(N, device=device, dtype=dtype) * (yaw_max - yaw_min) + yaw_min
    dr = torch.rand(N, device=device, dtype=dtype)*0.8 + 0.5

    half_dz = dz_max / 2
    low_margin_z = z - z_min
    high_margin_z = z_max - z
    alpha_low_z = torch.clamp(low_margin_z / half_dz, 0, 1)
    alpha_high_z = torch.clamp(high_margin_z / half_dz, 0, 1)
    dz_lo = -half_dz * alpha_low_z
    dz_hi = half_dz * alpha_high_z
    dz = torch.rand(N, device=device, dtype=dtype) * (dz_hi - dz_lo) + dz_lo

    half_dy = dyaw_max / 2
    low_margin_y = yaw - yaw_min
    high_margin_y = yaw_max - yaw
    alpha_low_y = torch.clamp(low_margin_y / half_dy, 0, 1)
    alpha_high_y = torch.clamp(high_margin_y / half_dy, 0, 1)
    dy_lo = -half_dy * alpha_low_y
    dy_hi = half_dy * alpha_high_y
    dyaw = torch.rand(N, device=device, dtype=dtype) * (dy_hi - dy_lo) + dy_lo

    out = torch.stack([r, phi, z, yaw, dr, dz, dyaw], dim=1)
    return out, center, normal




def is_interior(sample: torch.Tensor) -> bool:
    r, phi, z, yaw, dr, dz, dyaw = sample.unbind(0)

    ok_r = (r <= r_max)
    ok_phi = (phi >= phi_min) & (phi <= phi_max)
    ok_z = (z >= z_min) & (z <= z_max)

    yaw_lo = phi - delta_yaw 
    yaw_hi = phi + delta_yaw 
    ok_yaw = (yaw >= yaw_lo) & (yaw <= yaw_hi)

    ok_dr = (dr >= 0.3) & (dr <= 1.3)
    ok_dz = (dz >= -dz_max/2) & (dz <= dz_max/2)
    ok_dyaw = (dyaw >= -dyaw_max/2) & (dyaw <= dyaw_max/2)

    violations = []
    if not ok_r.item():    violations.append('r')
    if not ok_phi.item():  violations.append('phi')
    if not ok_z.item():    violations.append('z')
    if not ok_yaw.item():  violations.append('yaw')
    if not ok_dr.item():   violations.append('dr')
    if not ok_dz.item():   violations.append('dz')
    if not ok_dyaw.item(): violations.append('dyaw')

    
    print(f"Violations: {violations}")

    return bool(ok_r & ok_phi & ok_z & ok_yaw & ok_dr & ok_dz & ok_dyaw)

def is_inside_gate_region(center: torch.Tensor,
                          normal: torch.Tensor,
                          state: torch.Tensor,
                          r_in: float,
                          r_out: float,
                          epsilon: float = epsilon) -> bool:
    pos = state[:3]
    vec = center - pos
    along = torch.dot(normal, vec)
    if not (0 <= along <= epsilon):
        return False
    proj = vec - normal * along
    dist = torch.norm(proj)
    if dist > r_in:
        return False
    dz = state[4]
    if not (-0.2 <= dz <= 0.2):
        return False
    dr = state[3]
    if not (0.3 <= dr):
        return False
    dyaw = state[6]
    if not (-0.5 <= dyaw <= 0.5):
        return False
    return True

def sample_boundary_points(N, center, normal):
    interior, center, normal = sample_interior_points(N, center, normal)
    boundary = interior.clone()
    device = center.device
    for i in range(N):
        dims = torch.randperm(7, device=device)[:4]
        signs = torch.rand(4, device=device) < 0.5
        for j, dim in enumerate(dims):
            flip_min = signs[j]
            if dim == 0:
                boundary[i, 0] = r_max
            elif dim == 1:
                boundary[i, 1] = (phi_min) if flip_min else (phi_max)
            elif dim == 2:
                boundary[i, 2] = z_min if flip_min else z_max
            elif dim == 3:
                phi_i = boundary[i, 1]
                yaw_lo = phi_i - delta_yaw
                yaw_hi = phi_i + delta_yaw
                boundary[i, 3] = yaw_lo if flip_min else yaw_hi
            elif dim == 4:
                boundary[i, 4] = 0.6 if flip_min else 1.6
            elif dim == 5:
                boundary[i, 5] = (-dz_max/2) if flip_min else (dz_max/2)
            elif dim == 6:
                boundary[i, 6] = (-dyaw_max/2) if flip_min else (dyaw_max/2)
    return boundary, center, normal

