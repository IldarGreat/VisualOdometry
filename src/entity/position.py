import numpy as np


class Position:
    def __init__(self):
        self.current_pos = np.zeros((3, 1))
        self.current_rot = np.eye(3)
        pass

    def get_x(self):
        return self.current_pos[0][0]

    def get_y(self):
        return self.current_pos[2][0]

    def get_z(self):
        return self.current_pos[1][0]