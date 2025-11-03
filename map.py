import torch
import json
import os
from pathlib import Path
from config import DEVICE, DTYPE

class Ring:
    def __init__(self, center, normal, inner_radius, outer_radius):
        # Store center and normal as torch tensors on DEVICE
        self.center = torch.tensor(center, device=DEVICE, dtype=DTYPE)
        self.normal = torch.tensor(normal, device=DEVICE, dtype=DTYPE)
        norm = torch.norm(self.normal)
        if norm.item() < 1e-6:
            raise ValueError("Ring normal must be non-zero")
        self.normal = self.normal / norm

        # Radii remain Python floats
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)

class Map:
    def __init__(self, ring_file=None, rings=None):
        if rings is not None:
            self.rings = rings
        elif ring_file is not None:
            # Handle paths: if file doesn't exist, try parent directory (for scripts/)
            if not os.path.exists(ring_file):
                # Try parent directory (project root)
                project_root = Path(__file__).parent
                parent_path = project_root / ring_file
                if parent_path.exists():
                    ring_file = str(parent_path)
                else:
                    raise FileNotFoundError(f"Could not find {ring_file} in current directory or project root")
            with open(ring_file, 'r') as f:
                data = json.load(f)
            self.rings = []
            for entry in data:
                c    = entry["center"]
                n    = entry.get("orientation_diag", [0, 0, 1])
                rin  = entry["short_radius"]
                rout = entry["long_radius"]
                self.rings.append(Ring(c, n, rin, rout))
        else:
            self.rings = [
                Ring([0, 0, 10], [0, 0, 1], 4, 5)
            ]
