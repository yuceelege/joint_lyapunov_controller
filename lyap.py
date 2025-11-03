import torch
import torch.nn as nn

class LyapunovNet(nn.Module):
    def __init__(self, state_dim: int, xi_star: torch.Tensor, epsilon: float = 1e-3):
        super().__init__()
        # store xi_star as a (fixed) buffer
        self.register_buffer('xi_star', xi_star.view(-1).clone())
        self.R = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.epsilon = epsilon

    def forward(self, xi: torch.Tensor):
        delta = xi - self.xi_star.unsqueeze(0).expand_as(xi)
        RtR = self.R.t().mm(self.R)
        M = RtR + self.epsilon * torch.eye(RtR.size(0), device=RtR.device, dtype=RtR.dtype)
        # print(M)
        Md = delta.matmul(M)
        V = (Md * delta).sum(dim=-1)
        return V

class LyapunovNet2(nn.Module):
    def __init__(self, state_dim: int, xi_star: torch.Tensor, epsilon: float = 1e-3):
        super().__init__()
        # store xi_star as a (fixed) buffer
        self.register_buffer('xi_star', xi_star.view(-1).clone())
        self.R = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.epsilon = epsilon

    def forward(self, xi: torch.Tensor):
        delta = xi - self.xi_star#.to(xi.device) #.unsqueeze(0)#.expand_as(xi)
        RtR = self.R.t().mm(self.R)
        M = RtR + self.epsilon * torch.diag(torch.ones(RtR.size(0), device=RtR.device, dtype=RtR.dtype))
        # M = M.to(xi.device)
        # print(M)
        Md = delta.matmul(M)
        V = (Md * delta).sum(dim=-1, keepdim=True)
        return V


