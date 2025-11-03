import torch
import torch.nn as nn
from utils import euler_to_rot_matrix  # must support batched inputs
from config import DEVICE, DTYPE

class Environment(nn.Module):
    def __init__(self, agent, map_instance, focal_length=12.5, image_size=50, sharpness=100.0):
        super().__init__()
        self.agent        = agent
        self.map          = map_instance
        self.focal_length = focal_length
        self.image_size   = image_size
        self.sharpness    = sharpness  # sigmoid steepness

    def renderBatch(self, cur_state, cur_Rot=None):
        B      = cur_state.shape[0]
        device = cur_state.device
        dtype  = cur_state.dtype
        S      = self.image_size

        # prepare blank image
        image = torch.zeros((B, S, S, 3), device=device, dtype=dtype)

        # pixel grid
        coords = torch.arange(S, device=device, dtype=dtype)
        uu, vv = torch.meshgrid(coords, coords, indexing='xy')
        uu = uu[None].expand(B, -1, -1)
        vv = vv[None].expand(B, -1, -1)
        cx = cy = (S - 1) / 2

        # camera pose
        cam_pos = cur_state[:, :3]  # (B,3)
        if cur_Rot is not None:
            R_cw = cur_Rot           # (B,3,3)
        else:
            yaw   = cur_state[:, 5]
            zeros = torch.zeros_like(yaw)
            R_wc  = euler_to_rot_matrix(zeros, zeros, yaw)  # (B,3,3)
            R_cw  = R_wc.transpose(-2, -1)

        # world fallback axes
        world_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)[None].expand(B,3)
        alt_ref = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)[None].expand(B,3)

        β = self.sharpness
        for ring in self.map.rings:
            # ring parameters
            C     = torch.as_tensor(ring.center,        device=device, dtype=dtype)[None].expand(B,3)
            N     = torch.as_tensor(ring.normal,        device=device, dtype=dtype)[None].expand(B,3)
            r_in  = torch.as_tensor(ring.inner_radius,  device=device, dtype=dtype)
            r_out = torch.as_tensor(ring.outer_radius,  device=device, dtype=dtype)

            # ring center in camera frame
            diff = C - cam_pos
            cam  = torch.bmm(R_cw, diff.unsqueeze(-1)).squeeze(-1)  # (B,3)

            # replace torch.unbind by a crown supported functionAdd commentMore action
            # x_cam, y_cam, z_cam = cam.unbind(-1)
            # inv_x = 1.0 / (x_cam + 1e-6)
            x_cam = cam[..., 0]
            y_cam = cam[..., 1]
            z_cam = cam[..., 2]

            if x_cam>0:
                inv_x = 1.0 / (x_cam + 1e-6)
            else:
                inv_x = -1.0 / (-x_cam + 1e-6)

            u0 = cx - self.focal_length * (y_cam * inv_x)
            v0 = cy - self.focal_length * (z_cam * inv_x)

            # corrected fallback for near-vertical normals
            dot_nz = (N * world_z).abs().sum(-1, keepdim=True)

            # replace torch.where by a crown supported function
            #ref    = torch.where(dot_nz < 0.9, world_z, alt_ref)
            smooth_mask = torch.sigmoid(β * (0.9 - dot_nz))
            ref = smooth_mask * world_z + (1 - smooth_mask) * alt_ref

            # build ring-plane basis
            # e1 = torch.cross(ref, N, dim=-1)Add commentMore actions
            # e1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-6)
            e1 = torch.stack([
                ref[..., 1] * N[..., 2] - ref[..., 2] * N[..., 1],
                ref[..., 2] * N[..., 0] - ref[..., 0] * N[..., 2],
                ref[..., 0] * N[..., 1] - ref[..., 1] * N[..., 0],
            ], dim=-1)
            e1 = e1 /( torch.sqrt(torch.sum(e1 ** 2, dim=-1, keepdim=True)) + 1e-6)
            # e2 = torch.cross(N, e1, dim=-1)
            e2 = torch.stack([
                N[..., 1] * e1[..., 2] - N[..., 2] * e1[..., 1],
                N[..., 2] * e1[..., 0] - N[..., 0] * e1[..., 2],
                N[..., 0] * e1[..., 1] - N[..., 1] * e1[..., 0],
            ], dim=-1)
            e2 = torch.cross(N, e1, dim=-1)

            # rotate those axes into camera frame
            e1_cam = torch.bmm(R_cw, e1.unsqueeze(-1)).squeeze(-1)
            e2_cam = torch.bmm(R_cw, e2.unsqueeze(-1)).squeeze(-1)

            # project to image-plane directions
            d1 = torch.stack([
                -self.focal_length * e1_cam[:,1] * inv_x,
                -self.focal_length * e1_cam[:,2] * inv_x
            ], dim=-1)
            d2 = torch.stack([
                -self.focal_length * e2_cam[:,1] * inv_x,
                -self.focal_length * e2_cam[:,2] * inv_x
            ], dim=-1)

            # lengths = pixel radii; dirs = unit vectors
            # d1_len = d1.norm(dim=-1, keepdim=True)
            # d2_len = d2.norm(dim=-1, keepdim=True)

            d1_len = torch.sqrt(torch.sum(d1 ** 2, dim=-1, keepdim=True))
            d2_len = torch.sqrt(torch.sum(d2 ** 2, dim=-1, keepdim=True))
            d1_dir = d1 / (d1_len + 1e-6)
            d2_dir = d2 / (d2_len + 1e-6)

            a_out = r_out * d1_len.squeeze(-1)
            b_out = r_out * d2_len.squeeze(-1)
            a_in  = r_in  * d1_len.squeeze(-1)
            b_in  = r_in  * d2_len.squeeze(-1)

            # pixel offsets
            u_rel = uu - u0[:,None,None]
            v_rel = vv - v0[:,None,None]

            # project onto the true ellipse axes
            dot1 = d1_dir[:,0,None,None] * u_rel + d1_dir[:,1,None,None] * v_rel
            dot2 = d2_dir[:,0,None,None] * u_rel + d2_dir[:,1,None,None] * v_rel

            # implicit test
            val_out = (dot1 / (a_out[:,None,None] + 1e-6))**2 + \
                      (dot2 / (b_out[:,None,None] + 1e-6))**2
            val_in  = (dot1 / (a_in[:,None,None]  + 1e-6))**2 + \
                      (dot2 / (b_in[:,None,None]  + 1e-6))**2

            outer = torch.sigmoid(β * (1.0 - val_out))
            inner = torch.sigmoid(β * (val_in  - 1.0))
            geom_mask = outer * inner

            # back-face culling
            view_mask = torch.sigmoid(β * (x_cam - 1e-6))[:,None,None]
            ring_mask = geom_mask * view_mask

            # composite
            mask3 = ring_mask.unsqueeze(-1).expand(-1, S, S, 3)
            image = torch.max(image, mask3)

        return image
