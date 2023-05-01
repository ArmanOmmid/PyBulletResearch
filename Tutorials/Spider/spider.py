import sys
import os
import argparse

import gym
import pybullet
import pybullet_envs
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('type', type=str,
                    help='Type of Trial')
parser.add_argument('e', type=str,
                    help='Type of Trial')

def main(args):

    type = args.type
    train = type in ['train']
    test = type in ['test']

    env = gym.make("AntBulletEnv-v0")
    policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=[512, 512])
    model = PPO('MlpPolicy', env, learning_rate=0.0001, policy_kwargs=policy_kwargs, verbose=1)

    if train:
        MAX_AVERAGE_SCORE = 2000
        for i in range(10):
            env.render(mode="human")
            print("Training iteration: {}".format(i))
            model.learn(total_timesteps=100)
            model.save("_weights/PPO_Ant_Save")
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
            print("Mean Reward: {}".format(mean_reward))
            if mean_reward >= MAX_AVERAGE_SCORE:
                break
        
        del model

    if test:
        model.load("_weights/PPO_Ant_Load.zip")
        env.render(mode="human")
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)