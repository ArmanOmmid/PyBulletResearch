import gym
import numpy as np
import pybullet as p
class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):
        # Action space is in R^2 [0, -0.6], [1, 0.6] describing the throttle percentage and the steering agnles [-0.6, 0.6]
        """
        Action Space
        [0] Throttle
        [1] Steering Angle
        """
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, -0.6]),
            high=np.array([1, 0.6])
        )

        """
        Observation Space
        [0] x position of the car
        [1] y position of the car
        [2] x orientation of the car (unit)
        [3] y orientation of the car (unit)
        [4] x velocity of the car
        [5] y velocity of the var 
        [6] x position of the target we want to reach
        [7] y position of the target we want to reach
        """
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -1, -1, -5, -5, -10, -10]),
            high=np.array([10, 10, 1, 1, 5, 5, 10, 10])
        )

        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT) # We use p.DIRECT as we want to run our environment as quickly as possible when training a policy and only render when render() is called.

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]