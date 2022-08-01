#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import random

import gym
import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


# In[13]:


# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>Use CUDA</h2>

# In[14]:


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)


# <h2>Replay Buffer</h2>

# In[15]:


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, goal):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, goal)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, goal = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, goal
    
    def extend(self, trj):
        if len(self.buffer)+len(trj)<self.capacity:
            self.buffer.extend(trj)
        elif len(self.buffer)<self.capacity:
            rest_space=self.capacity-len(self.buffer)
            self.buffer.extend(trj[:rest_space])
            rest_trj=len(trj)-rest_space
            self.buffer[:rest_trj]=trj[rest_space:]
        else:
            self.buffer[self.position:self.position + len(trj)]=trj

        self.position=(self.position + len(trj)) % self.capacity

    
    def __len__(self):
        return len(self.buffer)

# In[16]:


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action


# In[17]:


# def plot(frame_idx, rewards):
#     clear_output(True)
#     plt.figure(figsize=(20,5))
#     plt.subplot(131)
#     plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
#     plt.plot(rewards)
#     plt.show()


# <h1>Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor</h1>
# <h2><a href="https://arxiv.org/abs/1801.01290">Arxiv</a></h2>

# In[18]:


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, goal):
        obs=np.concatenate((state, goal), axis=-1)
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        action  = action.detach().cpu().numpy()
        return action[0]


# In[19]:


def soft_q_update(batch_size, 
           gamma=0.99,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
          ):
    state, action, reward, next_state, done, goal = replay_buffer.sample(batch_size)

    state=np.concatenate((state, goal), axis=-1)
    next_state=np.concatenate((next_state, goal), axis=-1)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    

    expected_q_value = soft_q_net(state, action)
    expected_value   = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)


    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net(state, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = value_criterion(expected_value, next_value.detach())

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
    

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss  = std_lambda  * log_std.pow(2).mean()
    z_loss    = z_lambda    * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    soft_q_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


# In[20]:


from environment_SR import NAMOENV
# env = NormalizedActions(gym.make("Pendulum-v0"))
env=NAMOENV(use_gui=False)

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]
print("action_dim: {}".format(action_dim))
print("state_dim: {}".format(state_dim))
hidden_dim = 256

value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

save_path="models/SR_SAC/"
os.makedirs(save_path, exist_ok=True)
resume=True
if resume:
    value_net.load_state_dict(torch.load( save_path+"value_net.pth"))
    target_value_net.load_state_dict(torch.load(save_path+"target_value_net.pth"))
    soft_q_net.load_state_dict(torch.load(save_path+"soft_q_net.pth"))
    policy_net.load_state_dict(torch.load(save_path+"policy_net.pth"))
    print("------------------load model---------------------")

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = 500000
replay_buffer = ReplayBuffer(replay_buffer_size)


# In[21]:
max_eps     = 10000
max_steps   = 100
ep_idx      = 0
rewards     = []
batch_size  = 128
threshold   = 0.3
num_optim   = 5


def calc_r(st, dg, ag, threshold=0.3):
    d_sg=norm(dg-st)
    if d_sg<threshold:   # if achieved the final goal
        return 1, True
    else:
        d_sag=norm(ag-st)
        if d_sag<threshold:        # if reach the achieved goal
            return min(0,-d_sg+d_sag), True
        else:
            return min(0,-d_sg+d_sag), False

def sibling_rivalry(state, goal):
    trj=[]
    # get rollout
    for step in range(max_steps):
        action = policy_net.get_action(state, goal)
        next_state, reward, done, _ = env.step(action)
        
        trj.append((state, action, reward, next_state, done, goal))
        
        state = next_state
        
        if done:
            break

    # relabel
    relabel_trj=[]
    achieved_goal=trj[-1][0][-2:]   #treat the terminal state of rollout as antigoal

    eps_reward=0
    for item in trj:
        r, d=calc_r(item[0][-2:], goal, achieved_goal)
        relabel_trj.append((item[0], item[1], r, item[3], d, achieved_goal))
        eps_reward+=r

    return relabel_trj, eps_reward, done

# In[22]:
n_succ=0
n_total=0
success_cases=[0]*100

log_f=open(save_path+"log.txt","w")

episode=[]  # save episode for HER
while ep_idx < max_eps:
    state0, goal = env.reset()
    episode_reward = 0

    state=state0.copy()
    trj0, r0, d0=sibling_rivalry(state, goal)
    state=state0.copy()
    trj1, r1, d1=sibling_rivalry(state, goal)

    trj0_terminal_state=trj0[-1][0][-2:]
    trj1_terminal_state=trj1[-1][0][-2:]
    
    # print(trj0_terminal_state, goal)
    d_s0g=norm(trj0_terminal_state - goal)
    d_s1g=norm(trj1_terminal_state - goal)

    if d_s0g<d_s1g:
        episode_reward=r0
        trj_close=trj0
        trj_far=trj1
    else:
        episode_reward=r1
        trj_close=trj1
        trj_far=trj0

    if norm(trj0_terminal_state - trj1_terminal_state)<5 or d_s0g<3 or d_s1g<3:
        replay_buffer.extend(trj_close)
        replay_buffer.extend(trj_far)
    else:
        replay_buffer.extend(trj_far)
    
    success_cases[ep_idx%100]= 0
    if d0 or d1:
        success_cases[ep_idx%100]= 1
       
    rewards.append(episode_reward)

    if ep_idx % 4 == 1:
        temp="eps {}: reward={}, success_rate={}".format(ep_idx, rewards[-1], sum(success_cases)/50)
        print(temp)
        log_f.write(temp+"\r")
        
    for i in range(num_optim):
        if len(replay_buffer) > batch_size:
            soft_q_update(batch_size)  

    if ep_idx%500==1:
        torch.save(value_net.state_dict(), save_path+"value_net.pth")
        torch.save(target_value_net.state_dict(), save_path+"target_value_net.pth")
        torch.save(soft_q_net.state_dict(), save_path+"soft_q_net.pth")
        torch.save(policy_net.state_dict(), save_path+"policy_net.pth")
        print("------save model-----------")

    ep_idx+=1

log_f.close()
