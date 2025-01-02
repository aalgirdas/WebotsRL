import math
import sys
from controller import Motor, PositionSensor, Robot
import random
from controller import Supervisor
import time
import numpy as np
import os.path

import gymnasium as gym

import stable_baselines3

from stable_baselines3 import PPO , DQN
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box, Discrete


class RobotEnvironment(Supervisor, gym.Env ):
    def __init__(self):
        super().__init__()

        self.theta_threshold_radians = 0.3  
        self.x_threshold = 5 

        self.episode_reward = 0        


        self.timestep = int(self.getBasicTimeStep()*2)
        self.observation_space = Box(low=np.array([-10.4, -np.inf, -1.3, -np.inf]),  high=np.array([10.4, np.inf, 1.3, np.inf]),    dtype=np.float32)
        self.action_space = Box(low=np.array([-6.399]),  high=np.array([6.399]),    dtype=np.float32)


        self.sensors = []
        self.pendulum_sensor = self.getDevice('position sensor')
        self.pendulum_sensor.enable(self.timestep)
        self.sensors.append(self.pendulum_sensor)

        self.actuators = []
        for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel = self.getDevice(name)
            self.actuators.append(wheel)     


        self.robot = self.getSelf()

        #self.steps_per_episode = 200  # Max number of steps per episode
        #self.episode_score = 0  # Score accumulated during an episode
        #self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
        self.step_counter = 0
        self.epizod_counter = 0


        super().step(self.timestep)


    def get_observations(self):
        endpoint = self.getFromDef("POLE_ENDPOINT")
        cart_position = self.robot.getPosition()[0] # Position on x-axis
        cart_velocity = self.robot.getVelocity()[0] # Linear velocity on x-axis
        pole_angle = self.pendulum_sensor.getValue()  # Pole angle off vertical
        endpoint_velocity = endpoint.getVelocity()[3] # Angular velocity of endpoint . This function returns a vector containing exactly 6 values. The first three are respectively the linear velocities in the x, y and z direction. The last three are respectively the angular velocities around the x, y and z axes.

        return np.array([round(abs(cart_position)), cart_velocity, pole_angle, endpoint_velocity])


    def reset(self, seed=None, options=None):

        self.simulationResetPhysics()
        self.simulationReset()

        super().step(self.timestep)

        choice_tmp = random.choice([-2.5,-2.4,-2.3,-2.2,-2.1,-2.,2.,2.1,2.2,2.3,2.4,2.5])
        #choice_tmp = random.choice([-3.,  3.]) #   Angle: 6.26   imposible to make stable
        _, _, _,_,_ = self.step([choice_tmp])
        super().step(self.timestep*4)
        
        obs = self.get_observations()

        print(f"\n-----\n   Start Angle: {round(obs[2]* 180 / math.pi,2)}  Action: {choice_tmp}")

        self.episode_reward = 0

        info = {'some_info':'some_info'} 
        #return np.array([0, 0, 0, 0]).astype(np.float32) , info
        return obs.astype(np.float32) , info


    def step(self, action):
        #print(f'{action}')

        for wheel in self.actuators:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(action[0])
            
        super().step(self.timestep)

        self.step_counter = self.step_counter + 1

        observations = self.get_observations()

        done = bool(
            observations[0] < -self.x_threshold or
            observations[0] > self.x_threshold or
            observations[2] < -self.theta_threshold_radians or
            observations[2] > self.theta_threshold_radians or
            self.step_counter > 600
        )

        

        reward = 0 if done else 1- 2*abs(observations[2]) - abs(observations[0])/10  # Reward is given for every step taken, but the goal is to keep the pole upright and close to the center

        self.episode_reward += reward
        if done:
            self.epizod_counter = self.epizod_counter + 1
            #print(f'     epizod {self.epizod_counter:<4}   steps={self.step_counter:<4}     episode_reward={round(self.episode_reward,1):<5}   Last action: {round(action[0],1)}'   )  # Position={observations[0]:<5}    Angle={observations[2]:.2f} 
            print(f'     {self.step_counter:>60} steps                   epizod {self.epizod_counter:<40}       episode_reward={round(self.episode_reward,1):<5}   Last action: {action[0]}'   )  
            
            self.episode_reward = 0
            self.step_counter = 0

        return observations.astype(np.float32), reward, done, done, {}




env = RobotEnvironment()
check_env(env)

model_path =  r"Pioneer_RLmodel"
if os.path.exists(model_path+".zip"):
    print('The file exists.')
    model = PPO.load(model_path, env = env)
    #model.n_steps = 4096
    print(f"model: {model} Learning rate: {model.learning_rate}  Gamma (discount factor): {model.gamma}  Number of steps per update (n_steps): {model.n_steps}  Environment: {model.get_env()} Policy: {model.policy}")
else:
    print('The file does not exist.')
    model = PPO('MlpPolicy', env, n_steps=2048, verbose=1)



try_learn = False #   True  or False
if try_learn:
    env.reset()
    model.learn(total_timesteps = 10*1000000)
    
    model.save(model_path)

env.reset()

obs = env.get_observations()
print(f'                 Angle: {round(obs[2]* 180 / math.pi,2)} , obs: {obs} \n-------' )
#reward_all = 0
episode_number = 0
done = 0
for i in range(10000):

    if done == 1 or i == 0:
        print(f'episode: {episode_number}  Angle: {round(obs[2]* 180 / math.pi,2)}         ' ) # , obs: {obs}     i: {i}

    action = model.predict(obs)
    obs, reward, done, info, _ = env.step(action)
    #reward_all = reward_all + reward
    if done:
        #print(f'        i: {i}, episode_reward: {round(env.episode_reward,1)}       , obs: {obs}, reward: {reward}, done: {done}, info: {info} , action: {action}  \n-------' )
        print(f'\n-------' )
        obs, info = env.reset()
        #reward_all = 0
        episode_number += 1


env.simulationResetPhysics()
env.simulationReset()
env.simulationSetMode( env.SIMULATION_MODE_PAUSE)