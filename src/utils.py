import os

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


def get_odometry_information_from_config():
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    detector_name = config['odometry']['detector']
    descriptor_name = config['odometry']['descriptor']
    matcher_name = config['odometry']['matcher']
    opticalFlow_name = config['odometry']['opticalFlow']
    methodForComputingE_name = config['odometry']['methodForComputingE']
    return detector_name, descriptor_name, matcher_name, opticalFlow_name, methodForComputingE_name


def get_data_information_from_config():
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    camera_port = config['data']['liveCamera']
    video_path= config['data']['video']
    images_path = config['data']['images']
    ground_truth_path = config['data']['groundTruth']
    return camera_port, video_path, images_path, ground_truth_path

