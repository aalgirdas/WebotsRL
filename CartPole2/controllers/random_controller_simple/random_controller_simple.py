# https://www.generationrobots.com/media/Pioneer3AT-P3AT-RevA-datasheet.pdf?srsltid=AfmBOorohFQRJn4-ISxOJR9dC4lNgEONtMeEvDmi0E2YCkoqDpFK6BIu

import random
import sys
import os.path
import numpy as np

from controller import Robot 

timestep = 32  # Simulation time step in milliseconds. This means that the simulation is updated every 32 milliseconds
robot = Robot()

pendulum_sensor = robot.getDevice('position sensor')
pendulum_sensor.enable(timestep)


wheels = []
for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel = robot.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            wheels.append(wheel)



iter_num = 0
pole_angle_prev = 0
x_prev = 0
timestep = 128

while robot.step(timestep) != -1:  # Runs as long as robot.step(TIME_STEP) returns a non-negative value 
    iter_num += 1
    motor_speed = random.uniform(-6.4, 6.4)  # random motor speed
    
    for i in range(len(wheels)):  # For each wheel in the wheels list, the motor speed is set
        wheels[i].setVelocity(motor_speed)

    
    pole_angle = pendulum_sensor.getValue()
    pole_angle_velocity = (pole_angle - pole_angle_prev) / (timestep*0.001)
    pole_angle_prev = pole_angle
    x = x_prev + motor_speed*0.1* (timestep*0.001)
    x_prev = x


    if iter_num % 10 == 0:  # Prints the pole angle every 10 iterations
        print(f'pole_angle: {round(pole_angle,2):>5}  pole_angle_velocity: {round(pole_angle_velocity,2):>5}  x:{round(x,2):>5}  motor_speed: {round(motor_speed,2):>5}  after {iter_num} iterations   simulation time: {robot.getTime()}')
