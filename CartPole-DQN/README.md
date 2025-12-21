
# CartPole-v1 with Deep Q-Network (DQN)

A PyTorch implementation of Deep Q-Network (DQN) for solving the CartPole-v1 environment from OpenAI Gymnasium. This project demonstrates reinforcement learning fundamentals with experience replay, target networks, and epsilon-greedy exploration with cosine annealing.

## Overview

This implementation trains an agent to balance a pole on a moving cart using Deep Q-Learning. The agent learns through trial and error, gradually improving its policy to maximize the episode duration.

## Results

### Training Progress

| Stage | Performance | Video |
|-------|-------------|-------|
| **Initially** | Random exploration, poor performance | ![Initial Training](imgs/initially.mp4) |
| **Mid-Training** | Learning basic balancing strategies | ![Mid Training](imgs/mid-training.mp4) |
| **After Training** | Optimal policy, consistent high scores | ![After Training](imgs/after-training.mp4) |

## Architecture

### Network Design
```
Input (4) → Linear(128) → LeakyReLU 
         → Linear(256) → LeakyReLU
         → Linear(128) → LeakyReLU  
         → Linear(64) → LeakyReLU
         → Output(2)
```

**Key Components:**
- **Input**: 4 state variables (cart position, velocity, pole angle, angular velocity)
- **Output**: 2 Q-values (move left, move right)
- **Hidden Layers**: Progressive expansion and contraction (128→256→128→64)
- **Activation**: LeakyReLU (α=0.00001) for gradient stability

## Hyperparameters

```python
input_values: 4              # State space dimension
output_actions: 2            # Action space dimension
max_length: 250              # Replay buffer capacity
max_episode: 2000            # Total training episodes
n_moves: 140                 # Max steps per episode
learning_rate: 5e-3          # Adam optimizer learning rate
update_step: 750             # Target network update frequency
batch_size: 64               # Training batch size
gamma: 0.98                  # Discount factor
```

## Key Features

### 1. **Experience Replay Buffer**
- Stores transitions: `(state, action, reward, next_state, done)`
- Breaks temporal correlations in training data
- Maximum capacity: 250 transitions
- Enables efficient mini-batch learning

### 2. **Double Network Architecture**
- **Q-Train**: Actively learning network (updated every step)
- **Q-Target**: Stable target for TD updates (updated every 750 steps)
- Reduces overestimation bias and stabilizes training

### 3. **Cosine Annealing Exploration**
```python
eps = min_eps + 0.5 * (max_eps - min_eps) * (1 + cos(πt / T_max))
```
- Smooth decay from exploration (ε=1.0) to exploitation (ε=0.0001)
- Better than linear decay for finding optimal policies
- Balances exploration and exploitation throughout training

### 4. **Temporal Difference Learning**
```python
TD_target = reward + γ * max(Q_target(next_state)) * (1 - done)
Loss = MSE(Q_train(state, action), TD_target)
```

## Training Monitoring

### Weights & Biases Integration
- **Loss tracking**: Every 500 global steps
- **Evaluation rewards**: Every 1000 global steps (averaged over 5 episodes)
- **Episode rewards**: Logged after each episode

### Metrics Logged
- Training loss (MSE between Q-values and TD targets)
- Evaluation rewards (performance on greedy policy)
- Episode total rewards (training performance)

## Installation

```bash
# Clone the repository
git clone https://github.com/ajheshbasnet/reinforcement-learning-agents.git
cd reinforcement-learning-agents/CartPole-DQN

# Install dependencies
pip install torch numpy gymnasium wandb matplotlib tqdm
```

## Usage

### Training

```python
# Set your Weights & Biases API key
wandb.login(key="YOUR_API_KEY")

# Run training
python cartpole.py
```

The trained model will be saved as `weights.pt` after training completes.

### Evaluation

```python
# Load trained model
QTrain = DQNetwork(config).to(config.device)
QTrain.load_state_dict(torch.load("weights.pt"))

# Run evaluation with video recording
evaluation(QTrain)
```

Videos will be saved in the `videos/` directory.

## Project Structure

```
CartPole-DQN/
├── cartpole.py          # Main training script
├── weights.pt           # Saved model weights
├── videos/              # Recorded evaluation episodes
└── imgs/                # Training progression videos
    ├── initially.mp4
    ├── mid-training.mp4
    └── after-training.mp4
```

## Algorithm Details

### DQN Training Loop
1. **Observe** current state from environment
2. **Select** action using ε-greedy policy with cosine annealing
3. **Execute** action and observe reward, next state, done flag
4. **Store** transition in replay buffer
5. **Sample** random mini-batch from replay buffer
6. **Compute** TD target: `r + γ * max_a' Q_target(s', a')`
7. **Update** Q-train network using gradient descent on MSE loss
8. **Sync** Q-target network every 750 steps
9. **Repeat** until convergence

### Loss Function
```python
L(θ) = 𝔼[(Q(s,a;θ) - (r + γ * max_a' Q(s',a';θ⁻)))²]
```
where θ⁻ represents the frozen target network parameters.

## Expected Performance

- **Solved Criteria**: Average reward ≥ 195 over 100 consecutive episodes
- **Typical Training Time**: 500-1500 episodes
- **Max Episode Length**: 500 steps (environment limit)

## Customization

### Modify Network Architecture
```python
class DQNetwork(nn.Module):
    def __init__(self, config):
        # Customize layers here
        self.seq = nn.Sequential(...)
```

### Adjust Exploration Strategy
```python
def chooseAction(x, QTrain, t):
    # Modify epsilon calculation or use different schedule
    eps = your_custom_schedule(t)
```

### Change Hyperparameters
```python
@dataclass
class Config:
    # Modify any hyperparameter
    learning_rate: float = 1e-3
    gamma: float = 0.99
```

---

**Author**: [ajheshbasnet](https://github.com/ajheshbasnet)  
**Project**: [reinforcement-learning-agents](https://github.com/ajheshbasnet/reinforcement-learning-agents)
