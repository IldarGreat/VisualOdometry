import numpy as np
import yaml
import cv2

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
    video_path = config['data']['video']
    images_path = config['data']['images']
    ground_truth_path = config['data']['groundTruth']
    ground_truth_template = config['data']['groundTruthTemplate']
    return camera_port, video_path, images_path, ground_truth_path, ground_truth_template


def get_settings_information_from_config():
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    logs = config['settings']['logs']
    see_tracking = config['settings']['seeTracking']
    return logs, see_tracking

def init_odometry_params():
    detector_name, descriptor_name, matcher_name, opticalFlow_name, methodForComputingE_name = get_odometry_information_from_config()
    init_parameters = 0
    norm_type_for_BFM = None
    index_params_forFLANN, search_params_forFLANN = None, dict(checks=50)

    if descriptor_name == 'ORB':
        descriptor = cv2.ORB_create(3000)
        norm_type_for_BFM = cv2.NORM_HAMMING
        index_params_forFLANN = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        init_parameters += 1
    elif descriptor_name == 'SIFT':
        descriptor = cv2.SIFT_create(3000)
        norm_type_for_BFM = cv2.NORM_L2
        index_params_forFLANN = dict(algorithm=1, trees=5)
        init_parameters += 1
    elif descriptor_name == 'AKAZE':
        descriptor = cv2.AKAZE_create()
        norm_type_for_BFM = cv2.NORM_HAMMING
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
        detector = cv2.ORB_create(3000)
        init_parameters += 1
    elif detector_name == 'SIFT':
        detector = cv2.SIFT_create(3000)
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
