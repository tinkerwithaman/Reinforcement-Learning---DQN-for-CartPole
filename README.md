# Deep Q-Network (DQN) for CartPole

## Description
This project implements a Deep Q-Network (DQN) to solve the classic `CartPole-v1` environment from the `gymnasium` library. The goal is for an agent to learn a policy for balancing a pole on a cart by moving the cart left or right.

The agent uses a neural network to approximate the Q-value function and a replay buffer to store and sample past experiences, which helps stabilize the training process.

## Features
- DQN agent implemented in PyTorch.
- Experience Replay Buffer for efficient training.
- Epsilon-greedy strategy for balancing exploration and exploitation.
- Tracks and plots the total reward per episode.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd DQN_CartPole/
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the script:**
    ```bash
    python src/main.py
    ```

## Example Output
```
Episode 10, Total Reward: 25, Epsilon: 0.91
Episode 20, Total Reward: 15, Epsilon: 0.82
...
Episode 200, Total Reward: 158, Epsilon: 0.05
Episode 210, Total Reward: 199, Epsilon: 0.05
...
Episode 300, Total Reward: 200, Epsilon: 0.05
Training finished.
Plot of rewards per episode saved to assets/rewards_plot.png
```
*(A plot showing the rewards increasing over episodes will be saved in the `assets` folder.)*
