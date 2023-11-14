import numpy as np


class Position:
    def __init__(self):
        self.current_pos = np.zeros((3, 1))
        self.current_rot = np.eye(3)
        self.x_arr = []
        self.y_arr = []
        self.z_arr = []
        pass

    def update(self, R, t, scale):
        self.current_pos += self.current_rot.dot(t) * scale
        self.current_rot = R.dot(self.current_rot)
        self.x_arr.append(self.get_current_x())
        self.y_arr.append((-1)*self.get_current_y())
        self.z_arr.append((-1)*self.get_current_z())

    def get_current_x(self):
        return self.current_pos[0][0]

    def get_current_y(self):
        return self.current_pos[1][0]

    def get_current_z(self):
        return self.current_pos[2][0]

    def append_x(self, x):
        self.x_arr.append(x)

    def append_y(self, y):
        self.y_arr.append(y)

    def append_z(self, z):
        self.z_arr.append(z)

    def get_x_y_z(self, index):
        return self.x_arr[index], self.y_arr[index], self.z_arr[index]
