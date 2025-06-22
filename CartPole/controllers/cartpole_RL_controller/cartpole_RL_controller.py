import sys
from controller import Supervisor
import os.path

import gymnasium as gym
import numpy as np
import stable_baselines3

from stable_baselines3 import PPO , DQN
from stable_baselines3.common.env_checker import check_env

from gymnasium.spaces import Box, Discrete


def normalize_to_range(value, min_val, max_val, new_min, new_max, clip=False):
    value = float(value)
    min_val = float(min_val)
    max_val = float(max_val)
    new_min = float(new_min)
    new_max = float(new_max)

    if clip:
        return np.clip((new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max, new_min, new_max)
    else:
        return (new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max




class CartpoleRobot(Supervisor, gym.Env):
    def __init__(self):
        super().__init__()
        
        self.theta_threshold_radians = 0.3
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        ###self.observation_space = Box(low=np.array([0.4, -np.inf, -1.3, -np.inf]),  high=np.array([0.4, np.inf, 1.3, np.inf]),    dtype=np.float64)
        self.observation_space = Box(low=np.array([-10.4, -np.inf, -1.3, -np.inf]),  high=np.array([10.4, np.inf, 1.3, np.inf]),    dtype=np.float64)
        self.state = None
        # Define agent's action space using Gym's Discrete
        self.action_space = Discrete(2)

        self.timestep = int(self.getBasicTimeStep())
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.position_sensor = self.getDevice("polePosSensor")
        self.position_sensor.enable(self.timestep)

        self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        self.wheels = []
        for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.getDevice(wheel_name)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)

        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
        self.step_counter = 0
        self.epizod_counter = 0

    def get_observations(self):
        # Position on x-axis
        cart_position = normalize_to_range(self.robot.getPosition()[0], -10.4, 10.4, -1.0, 1.0)
        # Linear velocity on x-axis
        cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
        # Pole angle off vertical
        pole_angle = normalize_to_range(self.position_sensor.getValue(), -0.3, 0.3, -1.0, 1.0, clip=True)
        # Angular velocity y of endpoint
        endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)

        return [cart_position, cart_velocity, pole_angle, endpoint_velocity]



    def supervisor_step(self):
        super().step(self.timestep)

    def step(self, action):
        if action == 0:
            motor_speed = 5.0
        else:
            motor_speed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motor_speed)
        
        super().step(self.timestep)
        
        robot = self.getSelf()
        endpoint = self.getFromDef("POLE_ENDPOINT")
        ###self.state = np.array([robot.getPosition()[0], robot.getVelocity()[2], self.position_sensor.getValue(), endpoint.getVelocity()[3]])
        self.state = np.array(self.get_observations())
        
        done = bool( self.state[2] < -self.theta_threshold_radians or self.state[2] > self.theta_threshold_radians or self.step_counter > 1000  )
        
        reward = 0 if done else 1
        self.step_counter = self.step_counter + 1
        if done:
            self.epizod_counter = self.epizod_counter + 1
            print(f'epizod_counter = {self.epizod_counter}  steps number = {self.step_counter} ' )
            self.step_counter = 0

        return self.state.astype(np.float32), reward, done, done, {}  
        
        
          
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  ## AL
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.timestep)
        
        info = {'some_info':'some_info'}

        return np.array([0, 0, 0, 0]).astype(np.float32) , info
        
        
        


env = CartpoleRobot()
check_env(env)
print(f"{stable_baselines3.__version__=}")

cwd = os.getcwd()
print("Current working directory:", cwd)


#model_path =  r"C:\tmp\Simple_Webots_2actions_openAI_model"
model_path =  r"Simple_2actions_RL_model"
if os.path.exists(model_path+".zip"):
    print(f'loading model from: {model_path}')
    model = PPO.load(model_path, env = env)
    print(f"model: {model}")
else:
    print(f'The file {model_path} does not exist. Creating new model.')
    model = PPO('MlpPolicy', env, n_steps=2048, verbose=1)




model.learn(total_timesteps=2*1e4)


model.save(model_path)