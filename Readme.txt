Panda Robot Lift and pick and place Task with Reinforcement Learning

Requirements:

robosuite
stable-baselines3
numpy
gym

Features:

1. Custom reward function optimized for lift task
2. Automated model saving/loading
3. Training progress tracking
4. Model evaluation with visualization

Usage:
Run pick_and_pick_and_place_operation_for_final_project.py and choose from:

1. Continue training from latest model
2. Start fresh training
3. Evaluate latest model

Model Configuration:

1. SAC (Soft Actor-Critic) algorithm
2. Neural network: [1024, 1024, 512]
3. Batch size: 49,152
4. Learning rate: 5e-4
5. Control frequency: 180Hz

Reward Function:
Rewards for tasks:

1. Reaching the object
2. Successful grasping
3. Lifting behavior
4. Movement stability
5. Efficient actions