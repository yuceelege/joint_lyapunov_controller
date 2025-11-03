# agent.py
import torch
from config import DEVICE, DTYPE

class Drone:
    def __init__(self,
                 initial_state: torch.Tensor,  # [x, y, z, v_r, v_z, yaw, yaw_rate]
                 gravity: float = 9.81):
        # initial_state must be length-7 now
        self.state   = initial_state.to(DEVICE, dtype=DTYPE)
        self.gravity = gravity

    def _dynamics(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # state = [x, y, z, v_r, v_z, yaw, yaw_rate]
        # u     = [a_r, a_z, a_yaw]
        x, y, z, v_r, v_z, yaw, yaw_rate = state
        a_r, a_z, a_yaw                 = u

        dx = torch.zeros_like(state)
        # planar (polar) motion:
        dx[0] = v_r * torch.cos(yaw)      # ẋ
        dx[1] = v_r * torch.sin(yaw)      # ẏ
        dx[2] = v_z                       # ż

        # accelerations (double integrators):
        dx[3] = a_r                       # v̇_r
        dx[4] = a_z - self.gravity        # v̇_z (gravity down)
        dx[5] = yaw_rate                  # yaẇ
        dx[6] = a_yaw                     # yaẅ

        return dx

    def update_state(self, control_input: torch.Tensor, dt: float):
        # control_input = [a_r, a_z, a_yaw]
        u     = control_input.to(DEVICE, dtype=DTYPE)
        x     = self.state
        x_dot = self._dynamics(x, u)
        self.state = x + x_dot * dt

    def get_state(self) -> torch.Tensor:
        return self.state


class Drone2:
    def __init__(self,
                 gravity: torch.tensor = torch.tensor(9.81)):
        # initial_state must be length-7 now
        self.state   = None
        self.gravity = gravity

    def _dynamics(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Extract individual state components via slicing
        x     = state[..., 0]
        y     = state[..., 1]
        z     = state[..., 2]
        v_r   = state[..., 3]
        v_z   = state[..., 4]
        yaw   = state[..., 5]
        yaw_rate = state[..., 6]

        a_r   = u[..., 0]
        a_z   = u[..., 1]
        a_yaw = u[..., 2]

        # Compute derivatives using torch functions
        dx0 = v_r * torch.cos(yaw)             # ẋ
        dx1 = v_r * torch.sin(yaw)             # ẏ
        dx2 = v_z                              # ż
        dx3 = a_r                              # v̇_r
        dx4 = a_z  - self.gravity               # v̇_z
        dx5 = yaw_rate                         # yaẇ
        dx6 = a_yaw                            # yaẅ

    
        # Stack all derivatives into a tensor of shape (1, 7)
        dx = torch.stack([dx0, dx1, dx2, dx3, dx4, dx5, dx6], dim=-1)

        return dx

    def update_state(self, state: torch.Tensor(), control_input: torch.Tensor, dt: torch.Tensor):
        # control_input = [a_r, a_z, a_yaw]
        u     = control_input.to(DEVICE, dtype=DTYPE)
        x     = state.to(DEVICE, dtype=DTYPE)
        dt    = dt.to(DEVICE, dtype=DTYPE)
        x_dot = self._dynamics(x, u)
        self.state = x+x_dot*dt

    def get_state(self) -> torch.Tensor:
        return self.state
