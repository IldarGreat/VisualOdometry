import glob
import os

import cv2
import numpy as np
import yaml
from numpy import sort

from src.entity.position import Position


def get_camera_from_config():
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    calib_file = config['camera']['calibFile']
    if calib_file is not None and calib_file != 'None':
        with open('../data_set/images/kitti/00/calib.txt', 'r') as f:
            values = f.readline().split(" ")[1:]
            P = np.array([float(value) for value in values]).reshape((3, 4))
            K = P[0:3, 0:3]
    else:
        fx = config['camera']['fx']
        fy = config['camera']['fx']
        cx = config['camera']['fx']
        cy = config['camera']['fx']
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]])
    return K


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
    video_path = config['data']['video']
    images_path = config['data']['images']
    ground_truth_path = config['data']['groundTruth']
    ground_truth_template = config['data']['groundTruthTemplate']
    return camera_port, video_path, images_path, ground_truth_path, ground_truth_template


def get_settings_information_from_config():
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    see_tracking = config['settings']['seeTracking']
    return see_tracking


def init_odometry_params():
    detector_name, descriptor_name, matcher_name, opticalFlow_name, methodForComputingE_name = get_odometry_information_from_config()
    init_parameters = 0
    norm_type_for_BFM = None
    index_params_forFLANN, search_params_forFLANN = None, dict(checks=50)

    if descriptor_name == 'ORB':
        descriptor = cv2.ORB_create(3000)
        norm_type_for_BFM = cv2.NORM_L2
        index_params_forFLANN = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        init_parameters += 1
    elif descriptor_name == 'SIFT':
        descriptor = cv2.SIFT_create(3000)
        norm_type_for_BFM = cv2.NORM_L2
        index_params_forFLANN = dict(algorithm=1, trees=5)
        init_parameters += 1
    elif descriptor_name == 'AKAZE':
        descriptor = cv2.AKAZE_create()
        norm_type_for_BFM = cv2.NORM_L2
        init_parameters += 1
    elif descriptor_name == 'KAZE':
        descriptor = cv2.KAZE_create()
        norm_type_for_BFM = cv2.NORM_L2
        init_parameters += 1
    elif descriptor_name == 'BRISK':
        descriptor = cv2.BRISK_create(3000)
        norm_type_for_BFM = cv2.NORM_L2
        init_parameters += 1
    elif descriptor_name == 'SURF':
        descriptor = cv2.xfeatures2d.SURF_create(3000)
        norm_type_for_BFM = cv2.NORM_L2
        init_parameters += 1

    if detector_name == 'ORB':
        detector = cv2.ORB_create(1000)
        init_parameters += 1
    elif detector_name == 'SIFT':
        detector = cv2.SIFT_create(3000)
        init_parameters += 1
    elif detector_name == 'KAZE':
        detector = cv2.KAZE_create(3000)
        init_parameters += 1
    elif detector_name == 'AKAZE':
        detector = cv2.AKAZE_create()
        init_parameters += 1
    elif detector_name == 'BRISK':
        detector = cv2.BRISK_create()
        init_parameters += 1
    elif detector_name == 'FAST':
        detector = cv2.FastFeatureDetector_create(3000)
        init_parameters += 1
    elif detector_name == 'BLOB':
        detector = cv2.SimpleBlobDetector()
        init_parameters += 1
    elif detector_name == 'SURF':
        detector = cv2.xfeatures2d.SURF_create(3000)
        init_parameters += 1

    if matcher_name == 'BFMatcher':
        mather = cv2.BFMatcher(norm_type_for_BFM, crossCheck=True)
        init_parameters += 1
    elif matcher_name == 'FLANN':
        mather = cv2.FlannBasedMatcher(index_params_forFLANN, search_params_forFLANN)
        init_parameters += 1

    if opticalFlow_name == 'None':
        opticalFlow = opticalFlow_name
        init_parameters += 1

    if methodForComputingE_name == 'RANSAC':
        methodForComputingE = cv2.RANSAC
        init_parameters += 1
    elif methodForComputingE_name == 'LMEDS':
        methodForComputingE = cv2.LMEDS
        init_parameters += 1

    if init_parameters != 5:
        raise Exception(
            'One of odometry parameters not allowed {} {} {} {} {}'.format(detector_name, descriptor_name,
                                                                           matcher_name, opticalFlow_name,
                                                                           methodForComputingE_name))
    return descriptor, detector, mather, opticalFlow, methodForComputingE

def init_data_parameters():
    camera_port, video_path, images_path, ground_truth_path, ground_truth_template = get_data_information_from_config()
    data_init = 0
    cap, images, ground_truth = None, None, None
    if camera_port != 'None':
        cap = cv2.VideoCapture(camera_port)
        data_init += 1
        if cap is None or not cap.isOpened():
            raise Exception("Unable to open video source: {}".format(camera_port))

    if video_path != 'None':
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read(0)
            data_init += 1
            if not ret:
                raise Exception("Unable to read video file: {}".format(video_path))
        else:
            raise Exception("Unable to find video file: {}".format(video_path))

    if images_path != 'None':
        if os.path.exists(images_path):
            images = [cv2.imread(file) for file in sort(glob.glob(images_path + "/*.png"))]
            data_init += 1
        else:
            raise Exception("Unable to find images in: {}".format(images_path))

    if ground_truth_path != 'None':
        if os.path.exists(ground_truth_path):
            with open(ground_truth_path) as f:
                lines = f.readlines()
                ground_truth = Position()
                for line in lines:
                    if ground_truth_template == 'KITTI':
                        position = np.array(line.split()).reshape((3, 4)).astype(np.float32)
                        ground_truth.append_x(position[0, 3])
                        ground_truth.append_y(position[1, 3])
                        ground_truth.append_z(position[2, 3])
        else:
            raise Exception("Unable to find kitti in: {}".format(ground_truth_path))

    if data_init != 1:
        raise Exception("Only one data entry is required")

    return cap, images, ground_truth

def init_settings_parameters():
    see_tracking = get_settings_information_from_config()
    if see_tracking != 'None':
        see_tracking = True
    return see_tracking