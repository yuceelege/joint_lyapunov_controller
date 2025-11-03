import torch
import torch.nn.functional as F

from sampler import (
    r_min, r_max,
    phi_min, phi_max,
    z_min, z_max,
    delta_yaw,
    dr_max, dz_max, dyaw_max
)
def compute_threshold(lyap_net, boundary_samples, gamma=1.05):
    with torch.no_grad():
        V_vals = lyap_net(boundary_samples)
        return gamma * V_vals.min()


def violation_loss(lyap_net, f_cl_xi, xi, kappa, c0, rho, center, normal):
    Vp      = lyap_net(f_cl_xi)
    V       = lyap_net(xi)
   # print(V.item(),Vp.item())
    F_term  = Vp - (1 - kappa) * V
    H       = invariance_penalty(f_cl_xi)
   # print(H.item())
   # H = 0
    viol    = 100*F.relu(F_term) + H
   # print(100*F.relu(F_term).item())
    margin  = rho - V
   # print(margin.item())
    return F.relu(torch.min(viol, margin))



# def violation_loss(lyap_net, f_cl_xi, xi, kappa, c0, rho, center, normal):
#     Vp     = lyap_net(f_cl_xi)             # V(x⁺)
#     V      = lyap_net(xi)                  # V(x)
#     F_term = Vp - (1 - kappa) * V          # must be ≤ 0
#     H      = invariance_penalty(          # must be 0
#                f_cl_xi, center, normal)

#     # combined violation score (positive if any constraint is broken)
#     raw_viol = F.relu(F_term) + c0 * H     

#     # margin violation: raw_viol must be ≤ (ρ − V)
#     # so the amount by which it *exceeds* the margin is:
#     excess = raw_viol - (rho - V)          

#     # final loss: only penalize if excess > 0
#     return F.relu(excess)


def pgd_counterexample_step(xi, compute_loss_fn, alpha, xi_min, xi_max):
    xi = xi.clone().detach().requires_grad_(True)
    loss = compute_loss_fn(xi).sum()
    loss.backward()
    xi_next = xi + alpha * xi.grad
    xi_next = torch.max(torch.min(xi_next, xi_max), xi_min)
    return xi_next.detach()

def invariance_penalty(xi: torch.Tensor) -> torch.Tensor:
    r, phi, z, yaw, dr, dz, dyaw = xi.unbind(-1)
    phi_lo = phi_min 
    phi_hi = phi_max

    yaw_lo = phi - delta_yaw 
    yaw_hi = phi + delta_yaw

    pr = torch.relu(r_min - r) + torch.relu(r - r_max)
    pphi = torch.relu(phi_lo - phi) + torch.relu(phi - phi_hi)
    pz = torch.relu(z_min - z) + torch.relu(z - z_max)
    pyaw = torch.relu(yaw_lo - yaw) + torch.relu(yaw - yaw_hi)
    pdr = torch.relu(dr - dr_max)
    pdz = torch.relu(-dz_max/2 - dz) + torch.relu(dz - dz_max/2)
    pdyaw = torch.relu(-dyaw_max/2 - dyaw) + torch.relu(dyaw - dyaw_max/2)
   # print(pr + pphi + pz + pyaw + pdr + pdz + pdyaw)
    return (pr + pphi + pz + pyaw + pdr + pdz + pdyaw)*0.1

def project(x, center, normal):
    device, dtype = x.device, x.dtype

    r   = x[:, 0]
    phi = x[:, 1]
    z   = x[:, 2]
    yaw = x[:, 3]
    v   = x[:, 4:7]

    r_proj   = r.clamp(r_min, r_max)
    phi_proj = phi.clamp(phi_min, phi_max)
    z_proj   = z.clamp(z_min, z_max)

    yaw_min = phi_proj - delta_yaw + torch.pi
    yaw_max = phi_proj + delta_yaw + torch.pi
    yaw_proj = yaw.clamp(yaw_min, yaw_max)

    v_ranges = torch.tensor([dr_max, dz_max, dyaw_max],
                            device=device, dtype=dtype)
    v_proj = v.clamp(-v_ranges, v_ranges)

    return torch.cat([
        r_proj.unsqueeze(1),
        phi_proj.unsqueeze(1),
        z_proj.unsqueeze(1),
        yaw_proj.unsqueeze(1),
        v_proj
    ], dim=1)

def project_boundary(x, center, normal):
    phi_l = phi_min + torch.pi; phi_u = phi_max + torch.pi
    x_proj = x.clone()
    for xi in x_proj:
        xi[0] = xi[0].clamp(r_min, r_max)
        xi[1] = xi[1].clamp(phi_l, phi_u)
        xi[2] = xi[2].clamp(z_min, z_max)
        yl, yu = xi[1] - delta_yaw, xi[1] + delta_yaw
        xi[3] = xi[3].clamp(yl, yu)
        lo = torch.tensor([-dr_max,  -dz_max, -dyaw_max], device=xi.device, dtype=xi.dtype)
        hi = torch.tensor([ dr_max,  dz_max,  dyaw_max], device=xi.device, dtype=xi.dtype)
        xi[4:7] = xi[4:7].clamp(lo, hi)
        d = [
            (xi[0] - r_min).abs(), (r_max - xi[0]).abs(),
            (xi[1] - phi_l).abs(), (phi_u - xi[1]).abs(),
            (xi[2] - z_min).abs(), (z_max - xi[2]).abs(),
            (xi[3] - yl).abs(), (yu - xi[3]).abs(),
            (xi[4] + dr_max).abs(), (dr_max - xi[4]).abs(),
            (xi[5] + dz_max).abs(), (dz_max - xi[5]).abs(),
            (xi[6] + dyaw_max).abs(), (dyaw_max - xi[6]).abs()
        ]
        i = int(torch.argmin(torch.stack(d)))
        c, s = i // 2, i % 2
        if   c == 0: xi[0] = r_min if s == 0 else r_max
        elif c == 1: xi[1] = phi_l if s == 0 else phi_u
        elif c == 2: xi[2] = z_min if s == 0 else z_max
        elif c == 3: xi[3] = yl    if s == 0 else yu
        elif c == 4: xi[4] = -dr_max  if s == 0 else dr_max
        elif c == 5: xi[6] = -dz_max   if s == 0 else dz_max
        elif c == 6: xi[7] = -dyaw_max if s == 0 else dyaw_max
    return x_proj


