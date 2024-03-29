from distutils.log import info
import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
from environment_hard import NAMOENV

from PPO import PPO


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "env2_hard"
    has_continuous_action_space = True
    max_ep_len = 500           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0.1             # if required; add delay b/w frames

    total_test_episodes = 100   # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = NAMOENV(init_pos=[-4,0,0], goal_pos=[4,-1,0], use_gui=render)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = "2022-08-09-14-02-25"      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    # best_checkpoint_path="PPO_preTrained/explore/best_reward.pth"
    # ppo_agent.load(best_checkpoint_path)
    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    total_picks=[]
    finish_steps=[]

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        num_picks=0
        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, info = env.step(action)
            # print(action)
            ep_reward += reward

            num_picks+=len(info["pick"])

            # print(action)
            if render:
                # env.render_steps()
                env.render()
                time.sleep(frame_delay)

            if done:
                finish_steps.append(t)
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        
        total_picks.append(num_picks)

        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    success_rate=len(finish_steps)/total_test_episodes
    average_finish_steps=sum(finish_steps)/(len(finish_steps)+0.0001)
    average_picks=sum(total_picks)/(len(total_picks)+0.0001)
    print("sucess episodes: {}, success rate={}, average steps={}, average_picks={}".format(len(finish_steps),success_rate, average_finish_steps, average_picks))

    print("============================================================================================")


if __name__ == '__main__':

    test()
