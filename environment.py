# input: random distance, random final point speed
# output: time, consumed energy
# state: current speed, current remaining distance, time stamp
# action: throttle (acceleration demand), brake (deacceleration demand)
# state_size: 4
# action_size: 2
# future idea: gear shifting, traffic light timing, preceding vehicle


class env:
    def __init__(self):
        self.