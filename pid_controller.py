import torch
from config import DEVICE, DTYPE

class PIDController:
    """
    Stanley-style path follower with double-integrator yaw control:
    - Radial (forward) PID drives v_r toward projection of v_ref on path tangent.
    - Altitude PID drives vertical position to z_ref.
    - Stanley steering: outer PID loop on heading error to compute yaw_rate_ref (clamped),
      inner PD loop on yaw_rate error to compute yaw acceleration (double integrator).
    """
    def __init__(self,
                 Kp_r=1.5, Ki_r=0.3, Kd_r=2.5,
                 Kp_z=3.0, Ki_z=0.5, Kd_z=1.0,
                 Kp_th=3.0, Ki_th=0.2, Kd_th=0.5,
                 K_ct=1.0,
                 # yaw inner dynamics gains
                 Kp_yr=1.0, Kd_yr=0.5,
                 integrator_limit=1.0,
                 a_r_limit=3.0,
                 a_yaw_limit=1.0,
                 yaw_rate_limit=5.0,
                 gravity=9.81,
                 dt=0.1,
                 der_filter_alpha=0.8):
        # radial PID gains
        self.Kp_r, self.Ki_r, self.Kd_r = Kp_r, Ki_r, Kd_r
        # altitude PID gains
        self.Kp_z, self.Ki_z, self.Kd_z = Kp_z, Ki_z, Kd_z
        # heading outer PID gains
        self.Kp_th, self.Ki_th, self.Kd_th = Kp_th, Ki_th, Kd_th
        # Stanley lateral gain
        self.K_ct = K_ct
        # yaw-rate inner PD gains
        self.Kp_yr, self.Kd_yr = Kp_yr, Kd_yr

        # limits and parameters
        self.int_lim       = integrator_limit
        self.a_r_limit     = a_r_limit
        self.a_yaw_limit   = a_yaw_limit
        self.yaw_rate_limit= yaw_rate_limit
        self.gravity       = gravity
        self.dt            = dt
        self.alpha_r       = der_filter_alpha

        # internal buffers
        # radial
        self.de_r_filt  = torch.zeros((), device=DEVICE, dtype=DTYPE)
        self.I_r        = torch.zeros((), device=DEVICE, dtype=DTYPE)
        self.e_r_prev   = torch.zeros((), device=DEVICE, dtype=DTYPE)
        # altitude
        self.I_z        = torch.zeros((), device=DEVICE, dtype=DTYPE)
        self.e_z_prev   = torch.zeros((), device=DEVICE, dtype=DTYPE)
        # heading outer
        self.I_th       = torch.zeros((), device=DEVICE, dtype=DTYPE)
        self.e_th_prev  = torch.zeros((), device=DEVICE, dtype=DTYPE)
        # yaw-rate inner
        self.e_yr_prev  = torch.zeros((), device=DEVICE, dtype=DTYPE)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        # wrap to [-π, π]
        return (angle + torch.pi).remainder(2*torch.pi) - torch.pi

    def reset(self):
        # zero all internal buffers
        for buf in (self.de_r_filt, self.I_r, self.e_r_prev,
                    self.I_z, self.e_z_prev,
                    self.I_th, self.e_th_prev,
                    self.e_yr_prev):
            buf.zero_()

    def compute_control(self,
                        state: torch.Tensor,
                        p_ref: torch.Tensor,
                        v_ref: torch.Tensor,
                        tangent: torch.Tensor,
                        dref=None) -> torch.Tensor:
        """Compute control [a_r, a_z, a_yaw]"""
        # unpack state: x,y,z,v_r,v_z,theta,yaw_rate
        _, _, z, r, vz, theta, yaw_rate = state

        # -- Radial PID on speed along tangent --
        t_xy = tangent[:2] / (tangent[:2].norm() + 1e-8)
        r_ref = torch.dot(v_ref[:2], t_xy)
        e_r   = r_ref - r
        de_r  = (e_r - self.e_r_prev) / self.dt
        self.de_r_filt = self.alpha_r * self.de_r_filt + (1 - self.alpha_r) * de_r
        self.e_r_prev  = e_r
        self.I_r       = torch.clamp(self.I_r + e_r*self.dt,
                                      -self.int_lim, self.int_lim)
        a_r = self.Kp_r*e_r + self.Ki_r*self.I_r + self.Kd_r*self.de_r_filt
        a_r = torch.clamp(a_r, -self.a_r_limit, self.a_r_limit)

        # -- Altitude PID --
        e_z   = p_ref[2] - z
        de_z  = (e_z - self.e_z_prev) / self.dt
        self.e_z_prev = e_z
        self.I_z      = torch.clamp(self.I_z + e_z*self.dt,
                                      -self.int_lim, self.int_lim)
        a_z   = self.Kp_z*e_z + self.Ki_z*self.I_z + self.Kd_z*de_z + self.gravity

        # -- Stanley lateral (path) --
        n_xy   = torch.tensor([-t_xy[1], t_xy[0]], device=DEVICE, dtype=DTYPE)
        cte    = torch.dot(p_ref[:2] - state[:2], n_xy)
        yaw_path = torch.atan2(tangent[1], tangent[0])
        yaw_ref  = yaw_path + torch.atan(self.K_ct*cte/(r+1e-6))

        # -- Heading outer PID: compute desired yaw rate --
        e_th   = self._wrap_angle(yaw_ref - theta)
        de_th  = (e_th - self.e_th_prev) / self.dt
        self.e_th_prev = e_th
        self.I_th      = torch.clamp(self.I_th + e_th*self.dt,
                                      -self.int_lim, self.int_lim)
        yaw_rate_ref = (self.Kp_th*e_th + self.Ki_th*self.I_th + self.Kd_th*de_th)
        yaw_rate_ref = torch.clamp(yaw_rate_ref, -self.yaw_rate_limit, self.yaw_rate_limit)

        # -- Yaw-rate inner PD: compute yaw acceleration --
        e_yr  = yaw_rate_ref - yaw_rate
        de_yr = (e_yr - self.e_yr_prev) / self.dt
        self.e_yr_prev = e_yr
        a_yaw = self.Kp_yr*e_yr + self.Kd_yr*de_yr
        a_yaw = torch.clamp(a_yaw, -self.a_yaw_limit, self.a_yaw_limit)

        return torch.stack([a_r, a_z, a_yaw], dim=0)
