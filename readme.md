# Joint Vision-Based Controller and Lyapunov Function Training

This repository implements end-to-end vision-based control for autonomous drone navigation through ring sequences, combining imitation learning (DAgger) with Lyapunov-based safety verification. The system learns both a neural controller and a neural Lyapunov function jointly, ensuring stable and safe navigation behavior.

## Overview

This work extends the joint training method for controllers and Lyapunov functions (inspired by [Joint Training of Control Policies and Lyapunov Certificates](https://arxiv.org/pdf/2404.07956)) to **vision-based end-to-end learning** with a **differentiable renderer**. The key innovations are:

1. **Vision-based control**: CNN-based controller processes rendered camera images to output control commands
2. **Differentiable rendering**: End-to-end gradient flow from camera images through the controller
3. **Joint training**: Simultaneous learning of both the control policy and Lyapunov safety certificate
4. **Safety guarantees**: Learned Lyapunov function provides formal stability guarantees

## Features

- **End-to-end vision control**: Camera images → CNN → Control actions
- **Joint training pipeline**: Controller + Lyapunov function trained together
- **Standalone controller training**: DAgger-based imitation learning (no Lyapunov)
- **Differentiable environment**: Full gradient propagation through rendering
- **Safety verification**: Lyapunov-based stability certificates
- **Adversarial training**: PGD-based sampling for robust Lyapunov functions

## Training Modes

### 1. Joint Controller + Lyapunov Training (`lyapdag_trainer.py`)

Simultaneously learns both:
- **Vision-based controller**: CNN processes images to output control commands
- **Lyapunov function**: Neural network certifies stability and safety

This method alternates between:
- DAgger data collection from expert PID policy
- Adversarial sampling using PGD to find Lyapunov violations
- Joint optimization of both networks with safety-aware losses

```bash
python scripts/train/lyapdag_trainer.py --visualize
```

**Outputs**: 
- `weights/joint_controller.pth` - Trained vision-based controller
- `weights/joint_lyap.pth` - Trained Lyapunov function

### 2. Standalone Controller Training (`dagger_trainer.py`)

Traditional DAgger imitation learning that trains only the controller (no Lyapunov function). Useful for:
- Baseline comparisons
- Initial controller pretraining
- Systems where safety verification is not required

```bash
python scripts/train/dagger_trainer.py --visualize
```

**Outputs**:
- `weights/dagger_weights.pth` - Trained controller weights

## System Architecture

### Vision-Based Controller
- **Input**: 
  - Rendered camera image (50×50×3)
  - Velocity vector (4D: radial, vertical, yaw components)
  - Relative gate position (3D)
- **Architecture**: CNN (feature extraction) + MLP (control mapping)
- **Output**: Control actions `[a_r, a_z, a_yaw]` (radial acceleration, vertical acceleration, yaw rate)

### Lyapunov Function Network
- **Input**: Drone state `[x, y, z, v_r, v_z, yaw, yaw_rate]` (7D)
- **Output**: Scalar Lyapunov value V(x)
- **Properties**: Ensures V(x) decreases along trajectories, certifying stability

### Differentiable Renderer
- Perspective projection with sigmoid edges
- Batch rendering support for efficient training
- Full gradient flow enables end-to-end learning

### Drone Dynamics
- 7D state space: position, velocity, yaw, yaw rate
- Physics simulation with gravity
- Double integrator dynamics

## Repository Structure

```plaintext
├── config.py                # Centralized device/dtype configuration
├── agent.py                 # Drone dynamics & state update
├── controller.py            # CNN-based vision controller
├── pid_controller.py        # Expert PID controller (for DAgger)
├── environment.py           # Differentiable ring rendering
├── map.py                   # Ring & map definitions
├── lyap.py                  # Lyapunov network definitions
├── utils.py                 # Rotations, trajectories, utilities
├── utils2.py                # Lyapunov sampling, violation & projection
├── sampler.py               # State space sampling
├── sublevel_sampler.py      # Sublevel set sampling
├── scripts/
│   ├── train/                   # Training scripts
│   │   ├── dagger_trainer.py        # Standalone controller training
│   │   └── lyapdag_trainer.py       # Joint controller + Lyapunov training
│   └── test/                      # Testing scripts
│       ├── test_controller.py       # Closed-loop testing
│       ├── lyap_tester.py           # Lyapunov verification
│       └── propogate.py             # Closed loop propagation gradient check
├── weights/                 # Saved model weights (.pth files)
└── visualizations/          # Visualization tools
```

## Prerequisites

### Python Packages

Install required packages:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install torch torchvision numpy matplotlib scipy Pillow
```

### Core Dependencies

- **PyTorch** (≥1.9.0): Deep learning framework for neural networks
- **NumPy** (≥1.19.0): Numerical computing
- **Matplotlib** (≥3.3.0): Visualization and plotting
- **SciPy** (≥1.5.0): Scientific computing (spline interpolation)
- **Pillow** (≥8.0.0): Image processing
- **torchvision** (≥0.10.0): Image transforms and utilities

## Quick Start

1. **Install dependencies** (see [Prerequisites](#prerequisites) section):
   ```bash
   pip install -r requirements.txt
   ```

2. **Train joint controller + Lyapunov function**:
   ```bash
   python scripts/train/lyapdag_trainer.py --visualize
   ```

3. **Test the trained controller**:
   ```bash
   python scripts/test/test_controller.py --visualize
   ```

## Method Details

### Inspiration

This work is inspired by the joint training approach described in:
> **Joint Training of Control Policies and Lyapunov Certificates**  
> arXiv:2404.07956  
> [Paper Link](https://arxiv.org/pdf/2404.07956)

### Training Procedure

**Joint Training** (`lyapdag_trainer.py`):
1. Initialize controller and Lyapunov networks
2. For each episode:
   - Collect DAgger data: Expert PID policy generates (image, action) pairs
   - Find adversarial samples: PGD searches for states violating Lyapunov conditions
   - Optimize both networks: Joint loss includes control MSE + Lyapunov violation penalties
3. Safety-aware losses ensure controller respects Lyapunov certificate

**Standalone Training** (`dagger_trainer.py`):
1. Expert PID policy generates trajectories with mixed expert/learned actions (β-schedule)
2. Train controller to mimic expert via MSE loss
3. No Lyapunov component

## Testing & Verification

- **`test_controller.py`**: Closed-loop testing with visualization
- **`lyap_tester.py`**: Verify Lyapunov function properties
- **`propogate.py`**: One-step forward propagation analysis

