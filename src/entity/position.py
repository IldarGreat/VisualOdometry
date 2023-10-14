import numpy as np


class Position:
    def __init__(self):
        self.current_pos = np.zeros((3, 1))
        self.current_rot = np.eye(3)
        self.x_arr = []
        self.y_arr = []
        self.z_arr = []
        pass

    def update(self, R, t):
        self.current_pos += self.current_rot.dot(t)
        self.current_rot = R.dot(self.current_rot)
        self.x_arr.append(self.get_current_x())
        self.y_arr.append(self.get_current_y())
        self.z_arr.append(self.get_current_z())

    def get_current_x(self):
        return self.current_pos[0][0]

    def get_current_y(self):
        return self.current_pos[2][0]

    def get_current_z(self):
        return self.current_pos[1][0]
