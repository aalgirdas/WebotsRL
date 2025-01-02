from controller import Robot, Keyboard, Motion


class Pioneer3 (Robot):
    PHALANX_MAX = 8


    def startMotion(self, motion):
        # interrupt current motion
        if self.currentlyPlaying:
            self.currentlyPlaying.stop()

        # start new motion
        motion.play()
        self.currentlyPlaying = motion




    def printHelp(self):
        print('----------nao_demo_python----------')
        print('[<-][->]: left/right with speed 0.64')
        print('[A] [D]:  left/right with speed 6.40')
        print('')
        print('[H]: print this help message')

    def findAndEnableDevices(self):
        # get the time step of the current world.
        #self.timeStep = int(self.getBasicTimeStep())
        self.timeStep = int(32)
        print(self.timeStep)


        # keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timeStep)

    def __init__(self):
        Robot.__init__(self)
        self.currentlyPlaying = False

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__wheels = []
        self.__pendulum_sensor = None

        # Motors
        self.__wheels = []
        for name in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.__wheels.append(wheel)

        # initialize stuff
        self.findAndEnableDevices()
        self.printHelp()

    def run(self):


        # until a key is pressed
        key = -1
        while robot.step(self.timeStep) != -1:
            key = self.keyboard.getKey()
            if key > 0:
                break


        while True:
            key = self.keyboard.getKey()

            if key == ord('A'):
                for wheel in self.__wheels:
                    wheel.setVelocity(10*0.64)
            elif key == Keyboard.LEFT:
                for wheel in self.__wheels:
                    wheel.setVelocity(1*0.64)
            elif key == ord('D'):
                for wheel in self.__wheels:
                    wheel.setVelocity(10*-0.64)
            elif key == Keyboard.RIGHT:
                for wheel in self.__wheels:
                    wheel.setVelocity(1*-0.64)
            elif key == Keyboard.DOWN or key == ord('S'):
                for wheel in self.__wheels:
                    wheel.setVelocity(0)
            elif key == ord('H'):
                self.printHelp()

            if robot.step(self.timeStep) == -1:
                break
            
            #time.sleep(0.3)


# create the Robot instance and run main loop
robot = Pioneer3()
robot.run()
