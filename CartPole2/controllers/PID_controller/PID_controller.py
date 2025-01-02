import math
import sys
from controller import Motor, PositionSensor, Robot
import random
from controller import Supervisor
import time
import numpy as np

class RobotEnvironment(Supervisor ):
    def __init__(self):
        super().__init__()

        self.theta_threshold_radians = 0.3  
        self.x_threshold = 5 

        self.episode_reward = 0        


        self.timestep = int(self.getBasicTimeStep()*2)
        self.observation_space = (np.array([-10.4, -np.inf, -1.3, -np.inf]),  np.array([10.4, np.inf, 1.3, np.inf]))
        self.action_space = (np.array([-6.14]), np.array([6.14]))

        self.sensors = []
        self.pendulum_sensor = self.getDevice('position sensor')
        self.pendulum_sensor.enable(self.timestep)
        self.sensors.append(self.pendulum_sensor)

        self.actuators = []
        for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel = self.getDevice(name)
            #.setPosition(float('inf'))
            #wheel.setVelocity(0)
            self.actuators.append(wheel)     


        self.robot = self.getSelf()
        
        super().step(self.timestep)


    def get_observations(self):
        endpoint = self.getFromDef("POLE_ENDPOINT")
        cart_position = self.robot.getPosition()[0] # Position on x-axis
        cart_velocity = self.robot.getVelocity()[0] # Linear velocity on x-axis
        pole_angle = self.pendulum_sensor.getValue()  # Pole angle off vertical
        endpoint_velocity = endpoint.getVelocity()[3] # Angular velocity of endpoint . This function returns a vector containing exactly 6 values. The first three are respectively the linear velocities in the x, y and z direction. The last three are respectively the angular velocities around the x, y and z axes.

        return [cart_position, cart_velocity, pole_angle, endpoint_velocity]


    def reset(self, seed=None, options=None):

        self.simulationResetPhysics()
        self.simulationReset()
        self.episode_reward = 0

        super().step(self.timestep)

        _, _, _ = self.step(random.choice([-2, 2]))
        super().step(self.timestep*4)



    def step(self, action):
        for wheel in self.actuators:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(action)
            #print(f'                  {wheel}   {action}  ')  
        #print(action)
        super().step(self.timestep)

        observations = self.get_observations()

        done = bool(
            observations[0] < -self.x_threshold or
            observations[0] > self.x_threshold or
            observations[2] < -self.theta_threshold_radians or
            observations[2] > self.theta_threshold_radians
        )

        reward = 0 if done else 1

        self.episode_reward += reward


        return observations, reward, done


class PIDModel():
    
    def __init__(self, env):
        self.env = env

        self.P = 15#90
        self.I = 11#7
        self.D = 57#2

        self.previous_position = 0
        self.differential = 0
        self.integral_sum = 0

        self.best_episode_reward = 0

    def learn(self, total_timesteps = 200):
        for p_coef in range(1000, 0, -5):
            
            self.previous_position = 0
            self.differential = 0
            self.integral_sum = 0
            for i in range(total_timesteps):
                #print(f'learn: {i}')
                observations = self.env.get_observations()
                
                position = observations[2]
                self.integral_sum += position    
                self.differential = position - self.previous_position           
                            
                action = p_coef*position + self.I*self.integral_sum  + self.D*self.differential
                action = max(-6.4, min(action, 6.4)) 

                observations, reward, done = self.env.step(action)
                self.previous_position = position
                if done or i == total_timesteps-1:
                    self.P = p_coef if self.env.episode_reward > self.best_episode_reward else self.P
                    self.best_episode_reward = max(self.best_episode_reward, self.env.episode_reward)

                    print(f'{i}  {p_coef}   {self.env.episode_reward}    ---      {self.P}     {self.best_episode_reward}')

                    self.env.reset()
                    break

        self.best_episode_reward = 0            
        for d_coef in range(200, 0, -1):
            self.env.reset()
            self.previous_position = 0
            self.differential = 0
            self.integral_sum = 0
            for i in range(total_timesteps):
                observations = self.env.get_observations()
                
                position = observations[2]
                self.integral_sum += position    
                self.differential = position - self.previous_position           
                            
                action = self.P*position + self.I*self.integral_sum  + d_coef*self.differential
                action = max(-6.4, min(action, 6.4)) 

                observations, reward, done = self.env.step(action)
                self.previous_position = position
                if done or i == total_timesteps-1:
                    self.D = d_coef if self.env.episode_reward >= self.best_episode_reward else self.D
                    self.best_episode_reward = max(self.best_episode_reward, self.env.episode_reward)

                    print(f' --   {i}  {d_coef}   {self.env.episode_reward}    ---      {self.D}     {self.best_episode_reward}')

                    self.env.reset()
                    break

        self.best_episode_reward = 0            
        for i_coef in range(200, 0, -1):
            self.env.reset()
            self.previous_position = 0
            self.differential = 0
            self.integral_sum = 0
            for i in range(total_timesteps):
                observations = self.env.get_observations()
                
                position = observations[2]
                self.integral_sum += position    
                self.differential = position - self.previous_position           
                            
                action = self.P*position + i_coef*self.integral_sum  + self.D*self.differential
                action = max(-6.4, min(action, 6.4)) 

                observations, reward, done = self.env.step(action)
                self.previous_position = position
                if done or i == total_timesteps-1:
                    self.I = i_coef if self.env.episode_reward >= self.best_episode_reward else self.I
                    self.best_episode_reward = max(self.best_episode_reward, self.env.episode_reward)

                    print(f' ----   {i}  {i_coef}   {self.env.episode_reward}    ---      {self.I}     {self.best_episode_reward}')

                    self.env.reset()
                    break






    def predict(self, observations):

        position = observations[2]
        self.integral_sum += position    
        self.differential = position - self.previous_position
    
        action = self.P*position + self.I*self.integral_sum  + self.D*self.differential


        action = max(-6.4, min(action, 6.4))    

        self.previous_position = position

        return action






env = RobotEnvironment()
model = PIDModel(env)

try_learn = False # Set to True to learn the PID coefficients
if try_learn:
    env.reset()
    model.learn(total_timesteps = 800)

env.reset()
model.previous_position = 0
model.differential = 0
model.integral_sum = 0

for i in range(1000):
   
    observations = env.get_observations()
    action = model.predict(observations)
    observations, reward, done = env.step(action)

    print(f'predict: {i}  {action}   {model.P} {model.I} {model.D}                {[round(observation, 2) for observation in observations]}')
    if done:
        env.reset()
        break

