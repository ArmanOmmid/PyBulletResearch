import gym
import pybullet
import pybullet_envs
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make("AntBulletEnv-v0")

print("test")