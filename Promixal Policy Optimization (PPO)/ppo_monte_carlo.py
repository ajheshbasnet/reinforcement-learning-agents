import torch
import ale_py
import wandb
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

wandb.login(
    key = "PUT YOUR OWN API-KEY VIA OFFICIAL WEBSITE OF WANDB"
)

class configs:

  game_id = "RiverraidNoFrameskip-v4"  #YOU CAN CHANGE THE ATTARI GAME ID ACCORDING TO YOU
  max_step = 5_000
  stack_size = 4
  n_episodes = 100_000
  policy_lr = 3e-3
  value_lr = 2.5e-3
  discount_factor = 0.99
  epsilon = 0.2
  oldPolicy_updationStep = 2_000
  eval_steps = 5000
  cam_counter = 40_000
  eval_loops = 3
  device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = configs()

def create_run(configs):
    return wandb.init(
    name = "ppo",
    project="ppo",
    # Track hyperparameters and run metadata.
    config=vars(configs)
    )

def createEnvironment(cfg):

  env = gym.make(cfg.game_id, frameskip = 1, full_action_space=False, render_mode="rgb_array", max_episode_steps=configs.max_step)

  env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size = 84)
  # "scale_obs" means the pixels are scaled/normalised from 0 to 1 else it's in uint8 number--> keeping it False because to store it the float32 takes way huge memory so the training will be too much slow around 11s/iteration. Hence do it during the run time only.

  env = FrameStackObservation(env, cfg.stack_size)
  # it gives [frame(t-3), frame(t-2), frame(t-1), frame(t)] NOT [frame(t), frame(t+1), frame(t+2), frame(t+3)]

  # during env.reset() it gives obs = stack of [obs, obs, obs, obs] which is the same frame during the first time
  # so after the 1st action the stack becomes [f0, f0, f0, f1] and after another action it becomes [f0, f0, f1, f2] and so on.

  return env

env = createEnvironment(cfg)

class PolicyNetwork(nn.Module):

  def __init__(self, action_space):

    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(cfg.stack_size, 32, kernel_size=5, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=3),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU()
    )

    self.ffnn = nn.Sequential(
        nn.Linear(1024, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_space)
    )

  def forward(self, x):
    x = self.conv(x/255.)
    x = x.view(x.size(0), -1)
    x = self.ffnn(x)
    return x

  def log_probs(self, x):
    action_probs = torch.nn.functional.softmax(self(x), dim = -1)
    action_idx = torch.multinomial(action_probs, 1)

    log_prob = torch.gather(action_probs, -1, action_idx).log()
    return action_idx, log_prob

class Value_Network(nn.Module):

  def __init__(self):

    super().__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(cfg.stack_size, 32, kernel_size=5, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=3),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU()
    )

    self.ffnn = nn.Sequential(
        nn.Linear(1024, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

  def forward(self, x):
    x = self.conv(x/255.)
    x = x.view(x.size(0), -1)
    x = self.ffnn(x)
    return x

currentPolicy = PolicyNetwork(env.action_space.n).to(cfg.device)
oldPolicy = PolicyNetwork(env.action_space.n).to(cfg.device)
ValueNetwork = Value_Network().to(cfg.device)

oldPolicy.load_state_dict(currentPolicy.state_dict())
oldPolicy.eval()

policy_optimizer = torch.optim.Adam(currentPolicy.parameters(), cfg.policy_lr)
value_optimizer = torch.optim.Adam(ValueNetwork.parameters(), cfg.value_lr)

print(f'''
=======================================================================
> Actor-Net:  {sum(p.numel() for p in currentPolicy.parameters())/1e3} k
> Policy-Net: {sum(p.numel() for p in ValueNetwork.parameters())/1e3} k
-----------------------------------------------------------------------
> {cfg.device.upper()} is being used
=======================================================================
''')

def evaluationLoop(policynetwork, recordVideo = False):

  eval_env = createEnvironment(cfg)

  if recordVideo:
    eval_env = RecordVideo(
                           eval_env, video_folder="videos/",
                           episode_trigger=lambda episode_id: True, name_prefix="ppo"
                           )

  total_eval_rewards = 0
  total_eval_steps = 0

  with torch.no_grad():

    for _ in range(configs.eval_loops):

      obs, _ = eval_env.reset()
      done = False

      ep_reward = 0.0
      ep_step = 0

      while not done:

        action = policynetwork.log_probs(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(cfg.device))[0]
        next_obs,reward, terminated, truncated, _ =  eval_env.step(action.item())
        obs = next_obs
        ep_reward += float(reward)
        done = terminated or truncated
        ep_step += 1

      total_eval_rewards += ep_reward
      total_eval_steps += ep_step

    total_eval_rewards = total_eval_rewards / cfg.eval_loops
    total_eval_steps = int(total_eval_steps / cfg.eval_loops)

  eval_env.close()

  return total_eval_rewards, total_eval_steps

runs = create_run(cfg)

global_step = 0
wandblogin_step = 500

for steps in tqdm(range(cfg.n_episodes)):

  all_states_ = []
  all_log_probs_ = []
  all_rewards_ = []

  episodic_step = 0

  done = False

  state, _ = env.reset()

  training_reward = 0.0

  while not done:

    stateTensor = torch.tensor(state, dtype=torch.float32).to(cfg.device)

    action, log_prob = currentPolicy.log_probs(stateTensor.unsqueeze(0))

    next_state, reward, terminated, truncated, _ = env.step(action.item())

    done = terminated or truncated

    reward_tensor = torch.tensor(reward, dtype=torch.float).to(cfg.device)

    all_states_.append(stateTensor)
    all_log_probs_.append(log_prob)
    all_rewards_.append(reward_tensor)

    training_reward += float(reward)

    state = next_state

    runs.log({
        "global_step": global_step
    })

    episodic_step += 1
    global_step += 1

    if global_step%cfg.oldPolicy_updationStep==0:
      oldPolicy.load_state_dict(currentPolicy.state_dict())

    if global_step%cfg.eval_steps==0:

      if global_step%cfg.cam_counter==0:
        rec = True
      else:
        rec = False

      eval_reward, eval_steps = evaluationLoop(policynetwork=currentPolicy, recordVideo=rec)
      runs.log(
          {
              "avg-eval_rewards": eval_reward,
              "eval-episodic-step": eval_steps
          }
      )
      currentPolicy.train()

  all_states = torch.stack(all_states_)
  all_log_probs = torch.stack(all_log_probs_).view(-1, 1)
  all_rewards = torch.stack(all_rewards_).view(-1, 1)


  Gt = 0

  R = []

  for r in reversed(all_rewards):
    Gt = r + cfg.discount_factor * Gt
    R.insert(0, Gt)

  Rt = torch.stack(R).view(-1, 1)
  Vt = ValueNetwork(all_states)

  At = Rt - Vt.detach()

  At = (At - At.mean()) / (At.std() +1e-5)

  oldPolicyProb = oldPolicy.log_probs(all_states)[1]

  r = torch.exp(all_log_probs - oldPolicyProb)

  policy_loss = - torch.mean(torch.min(r * At, torch.clamp(r, 1 - cfg.epsilon, 1 + cfg.epsilon) * At))

  policy_optimizer.zero_grad()
  policy_loss.backward()
  policy_optimizer.step()

  value_loss = torch.nn.functional.mse_loss(Vt, Rt.detach())

  value_optimizer.zero_grad()
  value_loss.backward()
  value_optimizer.step()

  runs.log(
      {
      "episode-step" : episodic_step,
      "training-rewards": training_reward,
      "actor-loss": policy_loss.item(),
      "value-loss": value_loss.item()
      }
  )

evaluationLoop(currentPolicy, True)

