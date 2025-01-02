

import sys
from controller import Supervisor

import os.path

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO , DQN
from stable_baselines3.common.env_checker import check_env
import random



class OpenAIGymEnvironment(Supervisor, gym.Env):
    #def __init__(self, max_episode_steps=1000):
    def __init__(self):
        super().__init__()

        self.theta_threshold_radians = 0.3  
        self.x_threshold = 5 
        #high = np.array( [self.x_threshold * 2, np.finfo(np.float32).max,  self.theta_threshold_radians * 2,  np.finfo(np.float32).max ], dtype=np.float32  )
        self.action_space = gym.spaces.Discrete(17)  # {0, 1, 2, 3, 4 ... 16}
        #self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([-10.4, -np.inf, -1.3, -np.inf]),  high=np.array([10.4, np.inf, 1.3, np.inf]),    dtype=np.float32)

        self.state = None
        #self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        self.timestep = int(self.getBasicTimeStep())
        self.__wheels = []
        for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.__wheels.append(wheel)

        #self.pendulum_sensor = None
        self.pendulum_sensor = self.getDevice('position sensor')
        self.pendulum_sensor.enable(self.timestep)


        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)
        
        self.step_counter = 0
        self.epizod_counter = 0

        super().step(self.timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.timestep)

    #def supervisor_step(self):
    #    super().step(self.timestep)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  ## AL
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.timestep)

        #self.__wheels = []
        motor_speed = random.choice([-2, 2]) #random.randint(-2, 2)
        for wheel in self.__wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(motor_speed)

        #for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
        #    wheel = self.getDevice(name)
        #    wheel.setPosition(float('inf'))
        #    wheel.setVelocity(motor_speed)
        #    self.__wheels.append(wheel)

        super().step(self.timestep*4)

        # Sensors
        #self.pendulum_sensor = self.getDevice('position sensor')
        #self.pendulum_sensor.enable(self.timestep)

        #super().step(self.timestep*4)
        
        info = {'some_info':'some_info'}  ## AL

        # Open AI Gym generic
        return np.array([0, 0, 0, 0]).astype(np.float32) , info

    def step(self, action):

        for wheel in self.__wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(-6.4+0.8*action)  # -6.4+1.6*action    -3.2+0.8*action
        super().step(self.timestep)

        self.step_counter = self.step_counter + 1

        # Observation
        robot = self.getSelf()
        endpoint = self.getFromDef("POLE_ENDPOINT")
        #self.state = np.array([robot.getPosition()[2], robot.getVelocity()[2], self.pendulum_sensor.getValue(), endpoint.getVelocity()[3]])
        self.state = np.array([robot.getPosition()[0], robot.getVelocity()[2], self.pendulum_sensor.getValue(), endpoint.getVelocity()[3]])

        done = bool(
            self.state[0] < -self.x_threshold or
            self.state[0] > self.x_threshold or
            self.state[2] < -self.theta_threshold_radians or
            self.state[2] > self.theta_threshold_radians
        )

        # Reward
        reward = 0 if done else 1
        #reward = 0 if done else 1-np.absolute(self.state[2])
        
        if done:
            self.epizod_counter = self.epizod_counter + 1
            #print(f'epizod {self.epizod_counter:<4} Position={robot.getPosition():.2f}    Angle={self.state[2]:.2f}  steps={self.step_counter:<4} '   )
            poz_str=' '.join([str(round(num, 2)) for num in robot.getPosition()])
            print(f'epizod {self.epizod_counter:<4} Position={robot.getPosition()[0]:<5}    Angle={self.state[2]:.2f}  steps={self.step_counter:<4} '   )
            
            self.step_counter = 0

        
        return self.state.astype(np.float32), reward, done, done, {}




def main():
    env = OpenAIGymEnvironment()
    check_env(env)
    
    model_path =  r"Pioneer_17actions_model"
    if os.path.exists(model_path+".zip"):
        print('The file exists.')
        model = PPO.load(model_path, env = env)
        print(f"model: {model}")
    else:
        print('The file does not exist.')
        model = PPO('MlpPolicy', env, n_steps=2048, verbose=1)

    try_learn = True # Set to True to learn the PID coefficients
    if try_learn:
        env.reset()
        model.learn(total_timesteps=1*1e6)

        print('Training is finished, press `Y` for save model and replay...')
        env.wait_keyboard()
        model.save(model_path)


    obs = env.reset()
    obs = obs[0]
    print(obs)
    reward_all = 0
    episode_number = 0

    for i in range(3000):
        action, _states = model.predict(obs)
        obs, reward, done, info, _ = env.step(action)
        reward_all = reward_all + reward
        if reward_all == 1:
            print(f'episode_number: {episode_number} i: {i}, obs: {obs}, reward: {reward}, done: {done}, info: {info} , action: {action} , reward_all: {reward_all}  ' )
        if done:
            print(f'    i: {i}, obs: {obs}, reward: {reward}, done: {done}, info: {info} , action: {action} , reward_all: {reward_all}, episode_number: {episode_number} \n-------' )
            obs = env.reset()
            obs = obs[0]
            reward_all = 0
            episode_number += 1

    env.simulationResetPhysics()
    env.simulationReset()
    env.simulationSetMode( env.SIMULATION_MODE_PAUSE)

main()


