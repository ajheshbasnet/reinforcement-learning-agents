import torch
import random as random
import numpy as np
import torch.nn as nn
from collections import deque
from dataclasses import dataclass
import matplotlib.pyplot as plt
import gymnasium as gym
import wandb as wandb
import math
from tqdm import tqdm

wandb.login(key = "XXX")

def get_wandb_run(configs):
  run = wandb.init(
    name="CartPole-runs",
    config=vars(configs),
  )
  return run

@dataclass

class Config:
  input_values: int = 4
  output_actions: int = 2
  max_length: int = 250
  max_episode: int = 2000
  n_moves: int = 140
  learningRates: float = 5e-3
  updationStep: int = 750
  batch_size: int = 64
  device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

class DQNetwork(nn.Module):

    def __init__(self, config):
      super().__init__()
      self.config = config
      self.seq = nn.Sequential(
          nn.Linear(self.config.input_values, 128),
          nn.LeakyReLU(0.00001),

          nn.Linear(128,256),
          nn.LeakyReLU(0.00001),

          nn.Linear(256,128),
          nn.LeakyReLU(0.00001),

          nn.Linear(128,64),
          nn.LeakyReLU(0.00001),

          nn.Linear(64,self.config.output_actions)
      )

    def forward(self, x):
      return self.seq(x)

replay_buffer = deque(maxlen=config.max_length)

QTrain = DQNetwork(config).to(config.device)
QTarget = DQNetwork(config).to(config.device)

QTarget.load_state_dict(QTrain.state_dict())

def chooseAction(x, QTrain, t):
  max_eps = 1.0
  min_eps = 0.0001
  T_Max = config.n_moves * config.max_episode

  eps = min_eps + 0.5*(max_eps - min_eps) * (1 + torch.cos(torch.tensor((t*math.pi) / T_Max )))

  if random.random() < eps:   # exploration:
    return torch.randint(0, 2, (1,)).item()
  else:
    with torch.no_grad():
      output_states = QTrain(x)
      return torch.argmax(output_states, dim = -1).item()

def create_env(name: str):
  return gym.make(name)

def saveData(current_state, action, rewards, new_state, done):
  assert isinstance(state, np.ndarray), "it is not ndarray"
  assert isinstance(new_state, np.ndarray), "it is not ndarray"
  return replay_buffer.append(
      (current_state, action, rewards, new_state, done)
  )
env = create_env("CartPole-v1")

def get_batches(len: int):
  batches = random.sample(replay_buffer, len)
  current_state, action, rewards, new_state, done = zip(*batches)
  return current_state, action, rewards, new_state, done

run = get_wandb_run(config)

QTrainoptimizers = torch.optim.AdamW(QTrain.parameters(), lr = config.learningRates)

from gymnasium.wrappers import RecordVideo
import os

video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

def evaluation(QTrain, training: bool =  False):

  env = gym.make("CartPole-v1", render_mode="rgb_array")

  if training:
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True  # record every episode
    )

  obs, info = env.reset()
  done = False
  total_rewards = 0.0

  while not done:
    action = torch.argmax(QTrain(torch.tensor(obs, dtype=torch.float32).to(config.device)), dim = -1)
    obs, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated
    total_rewards += reward

  env.close()  # <-- REQUIRED
  return total_rewards

global_steps = 0
wandb_logging_step: int = 500
evaluation_step: int = 1000
gamma = 0.98
eval_itr = 5
t = 0

for episodes in tqdm(range(config.max_episode)):

  state, info = env.reset()
  total_rewards = 0.0

  for moves in range(config.n_moves):

    action = chooseAction(torch.tensor(state).to(config.device), QTrain, t)
    t += 1
    new_state, reward, terminated, truncation, _ = env.step(action)
    done = terminated or truncation

    total_rewards += reward       # type: ignore

    saveData(state, action, reward, new_state, done)
    state = new_state

    if done:
      break

    if len(replay_buffer) == config.max_length:
      batches = get_batches(config.batch_size)
      states, actions, rewards, new_states, dones = batches
      states, actions, rewards, new_states, dones = torch.tensor(np.stack(states), dtype = torch.float32 ).to(config.device), torch.tensor(actions, dtype=torch.long).to(config.device),torch.tensor(rewards,dtype=torch.float32).to(config.device), torch.tensor(np.stack(new_states), dtype = torch.float32).to(config.device),torch.tensor(dones, dtype=torch.bool).to(config.device)

      q_values = QTrain(states)  # [B, A]
      q_sa = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

      with torch.no_grad():
        q_vals = QTarget(new_states).max(1).values
        target = rewards + gamma * q_vals * (1 - dones.int())

      loss = torch.nn.functional.mse_loss(q_sa, target)

      QTrainoptimizers.zero_grad()
      loss.backward()
      QTrainoptimizers.step()

      global_steps +=1

      if global_steps%config.updationStep==0:
        QTarget.load_state_dict(QTrain.state_dict())

      if global_steps%wandb_logging_step:
        run.log({"loss": loss.item()})

      if global_steps%evaluation_step==0:
        total_evaluated = 0
        for _ in range(eval_itr):
          evaluated = evaluation(QTarget)
          total_evaluated += evaluated
        total_evaluated = total_evaluated/eval_itr
        run.log({"evaluation-reward": total_evaluated})

  run.log({"total_rewards": total_rewards})

run.finish()

torch.save(QTrain.state_dict(), "weights.pt")

QTarget = QTarget.to(config.device)

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os

video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

def evaluation(QTarget):

  env = gym.make("CartPole-v1", render_mode="rgb_array")

  env = RecordVideo(
      env,
      video_folder=video_dir,
      episode_trigger=lambda episode_id: True  # record every episode
  )

  obs, info = env.reset()
  done = False
  total_rewards = 0.0

  while not done:
    action = torch.argmax(QTrain(torch.tensor(obs, dtype=torch.float32).to(config.device)), dim = -1)
    obs, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated
    total_rewards += reward

  env.close()  # <-- REQUIRED
  return total_rewards

evaluation(QTrain)

