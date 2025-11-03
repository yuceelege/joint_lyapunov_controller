import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleControllerNN(nn.Module):
    def __init__(self, color=True, image_size=50):
        super().__init__()
        channels = 3 if color else 1
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=5, padding=2) 
        self.pool  = nn.MaxPool2d(kernel_size=2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        reduced = image_size // 2
        feat = 64 * reduced * reduced
        self.fc1 = nn.Linear(feat + 7, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, img, vel, rel_gate):
        x = F.relu(self.conv1(img)) # (batch_size, channels, image_size, image_size) -> (batch_size, 32, image_size, image_size)
        x = self.pool(x) # (batch_size, 32, image_size, image_size) -> (batch_size, 32, image_size//2, image_size//2)
        x = F.relu(self.conv2(x)) # (batch_size, 32, image_size//2, image_size//2) -> (batch_size, 64, image_size//2, image_size//2)
        x = x.flatten(start_dim=1) # (batch_size, 64, image_size//2, image_size//2) -> (batch_size, feat)
        x = torch.cat([x, vel, rel_gate], dim=1) # (batch_size, feat) -> (batch_size, feat+7)
        x = F.relu(self.fc1(x)) # (batch_size, feat+7) -> (batch_size, 128)
        return self.fc2(x) # (batch_size, 128) -> (batch_size, 3)

