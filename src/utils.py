import numpy as np
import yaml

def get_camera_from_config():
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    fx = config['camera']['fx']
    fy = config['camera']['fx']
    cx = config['camera']['fx']
    cy = config['camera']['fx']
    matrix = np.array([[fx, 0.0, cx],
                       [0.0, fy, cy],
                       [0.0, 0.0, 1.0]])
    return matrix