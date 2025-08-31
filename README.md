# Reinforcement-Learning-Based Pick-and-Place Task for Panda Robot

![banner](docs/figures/robosuite_banner.png)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Conda Environment](#2-create-a-conda-environment)
  - [3. Install PyTorch with CUDA Support](#3-install-pytorch-with-cuda-support)
  - [4. Install robosuite and Dependencies](#4-install-robosuite-and-dependencies)
- [Usage](#usage)
  - [1. Running the Script](#1-running-the-script)
  - [2. Training the Agent](#2-training-the-agent)
  - [3. Evaluating the Agent](#3-evaluating-the-agent)
- [Results](#results)
- [Notes](#notes)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview
This project leverages **robosuite**, a simulation framework for robotic manipulation, together with **Soft Actor–Critic (SAC)** to solve a **Pick-and-Place** task using a **Franka Panda** robot.  
The agent learns to autonomously pick up objects and place them in designated locations within a simulated environment.

---

## Features
- **Robosuite Integration**: Uses the `PickPlace` task with the Panda manipulator.
- **SAC**: Entropy-regularized RL for stable and sample-efficient learning.
- **Model Checkpointing**: Periodic saving of best and latest models.
- **GPU Support**: Configured to use CUDA for training acceleration.
- **Evaluation & Demo**: Run evaluation episodes and optional on-screen rendering.
- **Configurable Hyperparameters**: Adjust SAC + env hyperparameters via a single config.

---

## Prerequisites
- **Operating System**: Windows, macOS, or Linux  
- **Python**: 3.8+  
- **CUDA**: (Optional) For GPU acceleration, install a PyTorch build that matches your CUDA toolkit  
- **NVIDIA GPU**: Recommended for faster training

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SaiGopal-12/sac-pickplace-robosuite.git
cd sac-pickplace-robosuite
```

### 2. Create a Conda Environment
```bash
conda create -n robosuite_env python=3.10 -y
conda activate robosuite_env
```

### 3. Install PyTorch with CUDA Support
> Pick the command that matches your CUDA. Example for CUDA 11.8:
```bash
# CUDA 11.8 build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
> Or CPU-only:
```bash
pip install torch torchvision
```

### 4. Install robosuite and Dependencies
```bash
# robosuite + common deps
pip install robosuite gymnasium matplotlib pyyaml tqdm tensorboard

# if you use Stable Baselines3 (SB3) for SAC:
pip install stable-baselines3[extra]
```
> If your code needs extra packages, add them to `requirements.txt` and run:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Running the Script
```bash
# Example main entry (rename to your script name)
pick_and_pick_and_place_operation_for_final_project.py

```

When launched, you can present a simple menu like:
```
Choose an option:
1. Continue training from latest model (if available)
2. Start fresh training
3. Evaluate latest model
Enter your choice (1-3):
```

### 2. Training the Agent
**Option 1: Continue from latest model**  
Loads the most recent checkpoint from `trained_models/` and continues training.

**Option 2: Start fresh training**  
Begins training from scratch.

**Typical Training Parameters (example):**
- Total timesteps: 1,500,000 (adjustable)
- Batch size: 1024 (choose based on GPU memory)
- Replay buffer size: 8,000,000
- Learning rate: 5e-4 (actor/critic)
- Gamma: 0.95
- Entropy coefficient: Auto-tuned (initial value ~0.1)
- Train/eval cadence: `train_freq=3`, `gradient_steps=6`
- **Saving**:
  - Checkpoints every 5,000 timesteps → `trained_models/`
  - Final model saved at end of training
  - (Optional) Save VecNormalize stats for consistent evaluation

> Update these bullet values to exactly match your implementation.

### 3. Evaluating the Agent
**Option 3: Evaluate Latest Model**  
Runs N evaluation episodes (e.g., 5) and reports mean ± std of returns.  
Optionally runs **demonstration episodes** with on-screen rendering and **plots reward curves**.

**Evaluation Steps:**
- Load latest SAC checkpoint from `trained_models/`
- Run evaluation episodes (`--episodes 5` for example)
- (Optional) Render for 3 demo episodes
- (Optional) Plot rewards/metrics and save under `logs/` or `docs/figures/`

---

---

## Notes
- If Mujoco / robosuite complain about paths or renderers, follow robosuite’s install notes and ensure Mujoco is properly set up.
- Large artifacts (videos / checkpoints) should use **Git LFS** or be excluded via `.gitignore`.

---

## Acknowledgments
- [robosuite](https://robosuite.ai/) team for the simulation framework  
- SAC per *Haarnoja et al.*

---

## License
This project is released under the **MIT License**. See `LICENSE` for details.
