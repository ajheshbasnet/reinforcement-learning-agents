# PPO (Proximal Policy Optimization) - Atari Implementation

A clean, single-environment Monte Carlo implementation of PPO for Atari games using PyTorch and Gymnasium.

---

## Algorithm Overview

**Proximal Policy Optimization (PPO)** is a policy gradient method that strikes a balance between sample efficiency and implementation simplicity. It belongs to the family of **actor-critic** algorithms and has become one of the most popular RL algorithms due to its stability and effectiveness.

### Core Concept

PPO solves a fundamental challenge in policy gradient methods: **how to safely update the policy without taking steps that are too large**. Large policy updates can catastrophically degrade performance, while overly conservative updates learn slowly.

**The PPO Solution:**
- Maintains **two policies**: a `currentPolicy` (being actively trained) and an `oldPolicy` (frozen reference)
- Computes a **probability ratio** `r(θ) = π_new(a|s) / π_old(a|s)` between these policies for each action
- **Clips** this ratio to prevent the new policy from deviating too far from the old one
- This creates a "trust region" that keeps updates stable and prevents performance collapse

### The PPO Clipped Objective

```
L(θ) = E[ min(r(θ) · A, clip(r(θ), 1-ε, 1+ε) · A) ]
```

**Where:**
- **r(θ)** = π_currentPolicy(a|s) / π_oldPolicy(a|s) — ratio of action probabilities
- **A** = Advantage (how much better an action was compared to the value baseline)
- **ε** = clipping parameter (typically 0.2)
- **clip(r, 1-ε, 1+ε)** restricts r to [0.8, 1.2]

**Intuition:**
- If advantage A > 0 (good action): increase probability, but not by more than 20%
- If advantage A < 0 (bad action): decrease probability, but not by more than 20%
- This prevents destructive updates while still allowing meaningful learning

### Monte Carlo Returns

This implementation uses **Monte Carlo (MC)** estimation:
- Collects a **full episode** of experience before updating
- Computes actual returns `G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...` by working backwards from episode end
- More accurate than bootstrapping methods, but requires waiting for episode completion
- Well-suited for episodic tasks like Atari games

### Actor-Critic Architecture

**Two separate networks:**

1. **Policy Network (Actor)**: 
   - Outputs action probabilities for a given state
   - Learns which actions to take
   - Updated using the clipped PPO objective

2. **Value Network (Critic)**:
   - Estimates V(s), the expected future return from a state
   - Provides a baseline to reduce variance in policy updates
   - Updated using mean squared error between predicted and actual returns

### Training Flow

```
For each episode:
  1. Collect trajectory: (s_t, a_t, r_t, log_prob_t) using currentPolicy
  2. Compute returns: G_t for each timestep (Monte Carlo)
  3. Compute advantages: A_t = G_t - V(s_t)
  4. Normalize advantages (improves stability)
  5. Update currentPolicy using clipped PPO loss
  6. Update ValueNetwork using MSE loss
  7. Periodically sync oldPolicy ← currentPolicy
```

---

## Configuration Parameters

### Environment Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `game_id` | `"RiverraidNoFrameskip-v4"` | Atari game identifier (can be changed to any Atari game) |
| `max_step` | `5000` | Maximum steps per episode before truncation |
| `stack_size` | `4` | Number of consecutive frames stacked as input (provides temporal information) |

**Why frame stacking?** 
A single frame doesn't convey velocity or direction. Stacking 4 frames allows the agent to infer motion, crucial for games like Riverraid where you need to dodge moving obstacles.

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_episodes` | `100,000` | Total number of episodes to train |
| `policy_lr` | `3e-3` (0.003) | Learning rate for policy network (actor) |
| `value_lr` | `2.5e-3` (0.0025) | Learning rate for value network (critic) |
| `discount_factor` (γ) | `0.99` | Discount factor for future rewards (how much we value future vs immediate rewards) |
| `epsilon` (ε) | `0.2` | PPO clipping parameter (limits policy change to ±20%) |

**Learning rate choices:**
- Slightly higher for policy (3e-3) because we want the actor to explore and adapt
- Slightly lower for value (2.5e-3) because the critic should be more stable as a baseline

**Discount factor (0.99):**
- Close to 1.0 means we care about long-term rewards
- Essential for Atari games where you need to plan ahead (e.g., avoiding obstacles that will arrive in future frames)

### Update Frequencies

| Parameter | Value | Description |
|-----------|-------|-------------|
| `oldPolicy_updationStep` | `2000` | How often (in global steps) to update oldPolicy from currentPolicy |
| `eval_steps` | `5000` | How often to run evaluation episodes (without exploration) |
| `cam_counter` | `40,000` | How often to record video during evaluation |
| `eval_loops` | `3` | Number of episodes to average for evaluation metrics |

**Why update oldPolicy every 2000 steps?**
- Too frequent: policy ratio r(θ) stays near 1.0, limiting learning
- Too rare: old and current policies diverge too much, clipping becomes ineffective
- 2000 steps provides a good balance for Atari games

### System Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `device` | `"cuda"` or `"cpu"` | Automatically uses GPU if available, otherwise CPU |

---

## Network Architecture

### Policy Network (Actor)
```
Input: 4 × 84 × 84 (grayscale stacked frames)
  ↓
Conv2d(4→32, k=5, s=4) + ReLU
  ↓
Conv2d(32→64, k=4, s=3) + ReLU
  ↓
Conv2d(64→64, k=3, s=1) + ReLU
  ↓
Flatten to 1024
  ↓
Linear(1024→64) + ReLU
Linear(64→64) + ReLU
Linear(64→action_space)
  ↓
Output: Action logits → Softmax → Action probabilities
```

### Value Network (Critic)
```
Same convolutional layers as Policy Network
  ↓
Linear(1024→64) + ReLU
Linear(64→64) + ReLU
Linear(64→1)
  ↓
Output: Single value V(s)
```

**Architecture notes:**
- Convolutional layers extract spatial features from game frames
- Both networks share similar architecture but different final outputs
- Policy outputs action probabilities, Value outputs a single scalar
- Input normalization (x/255) converts pixel values from [0,255] to [0,1]

---

## Key Implementation Details

### Advantage Normalization
```python
At = (At - At.mean()) / (At.std() + 1e-5)
```
**Why?** Normalizing advantages to have mean=0 and std=1 prevents the scale of rewards from affecting learning dynamics. The 1e-5 prevents division by zero.

### Probability Ratio Computation
```python
r = torch.exp(all_log_probs - oldPolicyProb)
```
**Why logarithms?** Working in log-space provides numerical stability. `exp(log_π_new - log_π_old) = π_new / π_old`

### Frame Preprocessing
- **Grayscale**: Reduces input from 3 channels (RGB) to 1, speeding up training
- **Resize to 84×84**: Standard Atari preprocessing, balances detail with computational cost
- **Frame skip=4**: Agent only sees every 4th frame, matches human reaction time
- **Frame stacking**: Last 4 frames stacked to infer motion

---

## Logging and Evaluation

**Weights & Biases (wandb) Integration:**
- Tracks hyperparameters and metrics automatically
- Logs training rewards, losses, and evaluation performance
- Records videos of agent gameplay periodically

**Metrics tracked:**
- `training-rewards`: Episode return during training
- `actor-loss`: Policy network loss (clipped PPO objective)
- `value-loss`: Value network loss (MSE)
- `avg-eval_rewards`: Average reward over evaluation episodes
- `eval-episodic-step`: Average episode length during evaluation

---

## Usage

### Prerequisites
```bash
pip install torch gymnasium ale-py wandb
```

### Setup
1. Get your Weights & Biases API key from [wandb.ai](https://wandb.ai)
2. Replace the placeholder in the code:
```python
wandb.login(key="YOUR_API_KEY_HERE")
```

### Run Training
```bash
python ppo_atari.py
```

### Change Game
Modify the `game_id` in the `configs` class:
```python
game_id = "PongNoFrameskip-v4"  # or any other Atari game
```

Available games: Pong, Breakout, SpaceInvaders, MsPacman, Qbert, Seaquest, etc.

---

## Why This Implementation Works

**PPO's key advantages:**
1. **Stable**: Clipping prevents destructive policy updates
2. **Sample efficient**: Uses all collected data for updates (on-policy but reuses within epoch)
3. **Simple**: No complex trust region constraints like TRPO
4. **Effective**: State-of-the-art performance on many tasks

**Monte Carlo approach:**
- Accurate return estimates (no bootstrapping bias)
- Works well for episodic tasks
- Trade-off: requires full episode before learning (higher variance, lower bias)

---

## Expected Performance

- **Early training (0-10k steps)**: Random exploration, low rewards
- **Mid training (10k-50k steps)**: Agent learns basic game mechanics
- **Late training (50k+ steps)**: Agent develops strategies, rewards plateau

Training time depends on:
- Game complexity
- Hardware (GPU recommended)
- Episode length
- Hyperparameter tuning

---

## Common Issues & Solutions

**Problem: Training is slow**
- Solution: Ensure you're using GPU (`cuda`), reduce `eval_steps` frequency

**Problem: Rewards not improving**
- Solution: Try adjusting learning rates, epsilon value, or oldPolicy update frequency

**Problem: Training unstable**
- Solution: Decrease learning rates, ensure advantage normalization is working

---

## References

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

---

## License

MIT License - Feel free to use and modify for your projects!
