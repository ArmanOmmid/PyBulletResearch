import gymnasium as gym 
import numpy as np
import math
import pybullet as p
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human'], 'render_fps': 30}  
  
    def __init__(self, mode='DIRECT'):
        super().__init__()

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

        mode = p.DIRECT if mode == 'DIRECT' else p.GUI if mode == 'GUI' else None
        self.client = p.connect(mode) # We use p.DIRECT as we want to run our environment as quickly as possible when training a policy and only render when render() is called.
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

    def step(self, action):
        # Feed action to the car and get observation of car's state
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        self.prev_dist_to_goal = dist_to_goal

        # Done by running off boundaries
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or
                car_ob[1] >= 10 or car_ob[1] <= -10):
            self.done = True
        
        # Done by reaching goal
        elif dist_to_goal < 1:
            self.done = True
            reward = 50

        obs = np.array(car_ob + self.goal, dtype=np.float32)
        truncated = False
        info = dict()

        return obs, reward, self.done, truncated, info

    def reset(self, seed=None, options=None):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self.client)
        self.car = Car(self.client)

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        self.goal = (x, y)
        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)

        # Get observation to return
        car_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))
        
        obs = np.array(car_ob + self.goal, dtype=np.float32)
        info = dict()
        return obs, info

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]