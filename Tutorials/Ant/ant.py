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
                    help='Train or Test')
parser.add_argument('epochs', type=int,
                    help='Epochs')
parser.add_argument('--load', action='store_true',
                    help='Load')
parser.add_argument('--render', action='store_true',
                    help='Render')

def main(args):

    type = args.type
    train = type in ['train', 'both']
    test = type in ['test', 'both']

    epochs = args.epochs
    load = args.load
    render = args.render

    if not os.path.exists("_weights"):
        os.mkdir("_weights")

    env = gym.make("AntBulletEnv-v0")
    policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=[512, 512])
    model = PPO('MlpPolicy', env, learning_rate=0.0001, policy_kwargs=policy_kwargs, verbose=1)

    if load:
        model.load("_weights/PPO_Ant_Load.zip")

    if render:
        env.render(mode="human")

    if train:
        MAX_AVERAGE_SCORE = 2000
        for i in range(epochs):
            print("Training iteration: {}".format(i))
            model.learn(total_timesteps=10000)
            model.save("_weights/PPO_Ant_Save")
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
            print("Mean Reward: {}".format(mean_reward))
            if mean_reward >= MAX_AVERAGE_SCORE:
                break
        
        del model

    if train and test:
        os.rename("_weights/PPO_Ant_Save", "_weights/PPO_Ant_Load")

    if test:
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)