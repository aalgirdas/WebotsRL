# cartpole_PID.py

import math
import sys
from controller import Motor, PositionSensor, Robot
import random
from controller import Supervisor
import time

class CartpoleRobot(Supervisor ):
    def __init__(self):
        super().__init__()
        
        self.timestep = 64
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods

        self.wheels = []
        for wheelname in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:  # Wheel Initialization
            wheel = self.getDevice(wheelname)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
            self.wheels.append(wheel)
            
    def set_robot_speed(self, motor_speed):
        for i in range(len(self.wheels)):  # For each wheel in the wheels list, the motor speed is set
            self.wheels[i].setVelocity(motor_speed)    

    def get_observations(self):
        # Position on x-axis
        cart_position = self.robot.getPosition()[0]

        return cart_position


TIME_STEP = 64  # Simulation time step in milliseconds. It specifies how often the robot's main loop will be executed within the simulation. 
MAX_motor_speed = 25
robot = CartpoleRobot() # Robot()

position_sensor = robot.getDevice("polePosSensor")
position_sensor.enable(TIME_STEP)

previous_position = position_sensor.getValue()
previous_cart_position = 0
differential = 0
integral_sum = 0

cart_position = robot.get_observations()  # getPosition()[0]


i=0

while robot.step(TIME_STEP) != -1:  # Runs as long as robot.step(TIME_STEP) returns a non-negative value 
    position = position_sensor.getValue()
    integral_sum += position    
    differential = position - previous_position
    
    motor_speed = 700*position + 70*integral_sum  + 3*differential   
   
    motor_speed = min(MAX_motor_speed, max(-MAX_motor_speed, motor_speed))
    
    robot.set_robot_speed(motor_speed)

    previous_position = position
        
    cart_position = robot.get_observations()
    if abs(cart_position) > 4 :
        field = robot.robot.getField("translation")
        field.setSFVec3f([0.0, 0.0, .001])
        #children = robot.robot.getField("children")
        #pole = robot.robot.getField("POLE")
        #print(children)

    if(i%23==0):    
        print(f'{i:<5}     angle={round(position,2):>6}        cart_position={round(cart_position,2):>6}   motor={round(motor_speed,2):>6}    speed={round((cart_position-previous_cart_position)/(TIME_STEP*0.001),2):>6}') # {round(time.time() * 1000)}
    
    previous_cart_position = cart_position
    i+=1

