# Reinforcement Learning Agents

A focused collection of standard RL algorithms, each implemented as a clean Jupyter notebook (`.ipynb`) — built for understanding, not abstraction. Every notebook runs end-to-end, logs metrics to Weights & Biases, and is written so the algorithm speaks for itself.

---

## Philosophy

Most RL codebases bury the algorithm under layers of wrappers and utility classes. This repo does the opposite — each notebook is self-contained, linearly readable, and follows the same structure:

> **Intuition → Math → Implementation → Training → Results**

No base classes. No hidden logic. Just the algorithm.

---

## Implementations

| Algorithm | Environment | Notebook |
|-----------|-------------|----------|
| DQN | Atari Pong | `Pong-DQN/` |
| Double DQN | CartPole | `CartPole-Double-DQN/` |
| A2C | Gymnasium | `A2C/` |
| PPO | Continuous Control | `Proximal Policy Optimization (PPO)/` |
| Proximal PPO | Acrobot-v1 | `Proximal Policy Optimization - Acrobot/` |
| DDPG | Continuous Control | `DDPG/` |
| TD3 | Continuous Control | `TD3/` |
| REINFORCE + Baseline | Gymnasium | `Reinforce with Baseline (MC)/` |
| GRPO Fine-Tuning | LLM Fine-Tuning | `Group Relative Policy Optimization (GRPO)/` |

---

## W&B Logging

All notebooks log training metrics (rewards, losses, episode lengths) directly to [Weights & Biases](https://wandb.ai).

**Setup — one step:**

```python
import wandb
wandb.login(key="YOUR_WANDB_API_KEY")  # Get your key at https://wandb.ai/authorize
```

Replace `YOUR_WANDB_API_KEY` with your key from [wandb.ai/authorize](https://wandb.ai/authorize). That's it — metrics stream automatically once training starts.

Each notebook initializes its own `wandb.init(project=..., config=...)` run. You'll see reward curves, loss plots, and hyperparameter sweeps live in your W&B dashboard.

---

## Running a Notebook

Each directory is self-contained. Navigate to any algorithm folder and open the `.ipynb`:

```bash
cd "Proximal Policy Optimization (PPO)"
jupyter notebook ppo.ipynb
```

Install dependencies:

```bash
pip install torch gymnasium wandb numpy matplotlib
```

> Some environments (Atari, MuJoCo) need additional setup — see the `README` inside each subdirectory.

---

## Repository Structure

```
reinforcement-learning-agents/
├── CartPole-Double-DQN/
├── Pong-DQN/
├── A2C/
├── DDPG/
├── TD3/
├── Proximal Policy Optimization (PPO)/
├── Proximal Policy Optimization - Acrobot/
├── Reinforce with Baseline (MC)/
├── Group Relative Policy Optimization (GRPO)/
└── README.md
```

---

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Weights & Biases (`wandb`)
- NumPy, Matplotlib

---

## Contributing

New implementations should follow the same notebook format: intuition first, then derivation, then clean code. Include W&B logging and a short results section at the end of the notebook.

---

## References

Implementations follow the original papers. Citations are included at the top of each notebook.
