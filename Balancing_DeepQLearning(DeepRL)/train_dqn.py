import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy as np
import time
import random
import os
from cartpole_env import make_cart_pole_env, configure_pybullet
from replay_buffer import ReplayBuffer
from q_network import QNetwork


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--save_dir', type=str, default='models',
                        help="the root folder for saving the checkpoints")
    parser.add_argument('--gui', action='store_true', default=False,
                        help="whether to turn on GUI or not")
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    if not args.seed:
        args.seed = int(time.time())
    return args


def train_dqn(env, args, device):
    # set up seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    # ---
    # Write your code here to train DQN
    # ---
    
    target_nn = QNetwork(env).to(device)
    action_nn = QNetwork(env).to(device)
    nnbuffer = ReplayBuffer(10000)
    target_nn.load_state_dict(action_nn.state_dict())
    target_nn.eval()
    optimizer = optim.Adam(action_nn.parameters())
    loss_fn = nn.MSELoss()
    gamma = 0.999
    numeps = 800
    numsteps = 500
    epsend = 0.05
    epsstart = 0.9
    epsdecay = 200
    numbatch = 32
    updatenum = 5
    for eps in range(numeps):
       state = env.reset()
       episode_reward = 0
       for global_step in range(numsteps):
          epsilon = epsend + (epsstart - epsend) * \
	  np.exp(-1. * global_step / epsdecay)
          if random.random() < epsilon:
             action = env.action_space.sample()
          else:
              logits = action_nn.forward(state.reshape((1,)+state.shape), device)
              action = torch.argmax(logits, dim=1).tolist()[0]
          next_state, reward, done, _ = env.step(action)
          episode_reward += reward
          nnbuffer.put((state, action, reward, next_state, done))
          state = next_state
          if global_step > numbatch:
              s_state, s_actions, s_rewards, s_next_states, s_dones = nnbuffer.sample(numbatch)
              with torch.no_grad():
                  target_max = torch.max(target_nn.forward(s_next_states, device), dim=1)[0]
                  newval = torch.Tensor(s_rewards).to(device) + gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
              expval = action_nn.forward(s_state, device).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
              loss = loss_fn(newval, expval)
              optimizer.zero_grad()
              loss.backward()
              for param in action_nn.parameters():
                    param.grad.data.clamp_(-1, 1)
              optimizer.step()
              if done:
                 break
          if global_step % updatenum == 0:
              target_nn.load_state_dict(action_nn.state_dict())

          if next_state is None:
             break
       print(episode_reward)
       model_folder_name = f'episode_{eps:06d}_reward_{round(episode_reward):03d}'
       if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
             os.makedirs(os.path.join(args.save_dir, model_folder_name))
       torch.save(action_nn.state_dict(), os.path.join(args.save_dir, model_folder_name, 'q_network.pth'))
       print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "q_network.pth")}\n')

        

if __name__ == "__main__":
    args = get_args()
    configure_pybullet(rendering=args.gui, debug=True)
    env = make_cart_pole_env()
    device = torch.device('cpu')

    train_dqn(env, args, device)
