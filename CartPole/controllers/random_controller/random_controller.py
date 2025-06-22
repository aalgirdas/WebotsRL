from controller import Robot
import random

TIME_STEP = 64  # Simulation time step in milliseconds. It specifies how often the robot's main loop will be executed within the simulation. 
robot = Robot()
wheels = []
wheelsNames = ['wheel1', 'wheel2', 'wheel3', 'wheel4']



for wheelname in wheelsNames:  # Wheel Initialization
    wheel = robot.getDevice(wheelname)
    wheel.setPosition(float('inf'))
    wheel.setVelocity(0.0)
    wheels.append(wheel)



while robot.step(TIME_STEP) != -1:  # Runs as long as robot.step(TIME_STEP) returns a non-negative value 
    motor_speed = random.randint(-10, 10)  # random motor speed
    
    for i in range(len(wheels)):  # For each wheel in the wheels list, the motor speed is set
        wheels[i].setVelocity(motor_speed)


        