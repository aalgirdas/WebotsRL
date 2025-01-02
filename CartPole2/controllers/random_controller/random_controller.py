import random
import sys
import os.path
import numpy as np

from controller import Supervisor



class RobotEnvironment(Supervisor):
    def __init__(self):
        super().__init__()

        self.theta_threshold_radians = 0.3  
        self.x_threshold = 10 


        self.timestep = int(self.getBasicTimeStep())
        self.observation_space = (np.array([-10.4, -np.inf, -1.3, -np.inf]),  np.array([10.4, np.inf, 1.3, np.inf]))
        self.action_space = (np.array([-6.14]), np.array([6.14]))

        self.sensors = []
        self.pendulum_sensor = self.getDevice('position sensor')
        self.pendulum_sensor.enable(self.timestep)
        self.sensors.append(self.pendulum_sensor)

        self.actuators = []
        for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel = self.getDevice(name)
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
        super().step(self.timestep)


    def step(self, action):
        for wheel in self.actuators:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(action)
        super().step(self.timestep)

        observations = self.get_observations()

        done = bool(
            observations[0] < -self.x_threshold or
            observations[0] > self.x_threshold or
            observations[2] < -self.theta_threshold_radians or
            observations[2] > self.theta_threshold_radians
        )

        reward = 0 if done else 1


        return observations, reward, done



class RandomModel():
    
    def __init__(self, env):
        self.env = env

    def learn(self, total_timesteps = 200):
        for i in range(total_timesteps):
            print(f'learn: {i}')
            action = random.uniform(self.env.action_space[0][0], self.env.action_space[1][0])
            observations, reward, done = self.env.step(action)
            if done:
                self.env.reset()
                

    def predict(self, observations):
        action = random.uniform(self.env.action_space[0][0], self.env.action_space[1][0])
        return action




env = RobotEnvironment()
model = RandomModel(env)

env.reset()

model.learn(total_timesteps = 300)


for i in range(100):
    print(f'predict: {i}')
    
    observations = env.get_observations()
    action = model.predict(observations)
    env.step(action)