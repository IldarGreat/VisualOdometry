import glob
import os.path
import random
from copy import copy

import cv2
from matplotlib import pyplot as plt
from numpy import sort

from src.entity.camera import Camera
from src.entity.feature import Feature
from src.entity.position import Position
from src.utils import *


class IndirectVisualOdometry:

    def __init__(self):
        self.camera = Camera(get_camera_from_config())

        self.detector, self.descriptor, self.mather, self.opticalFlow, self.methodForComputingE = None, None, None, None, None
        self.init_odometry_parameters()

        self.cap, self.images, self.ground_truth_path = None, None, None
        self.init_data_parameters()

        self.position = Position()
        self.previous_feature = Feature()
        self.current_feature = Feature()

        pass

    def init_data_parameters(self):
        camera_port, video_path, images_path, ground_truth_path = get_data_information_from_config()
        data_init = 0
        if camera_port != 'None':
            self.cap = cv2.VideoCapture(camera_port)
            data_init += 1
            if self.cap is None or not self.cap.isOpened():
                raise Exception("Unable to open video source: {}".format(camera_port))

        if video_path != 'None':
            if os.path.exists(video_path):
                self.cap = cv2.VideoCapture(video_path)
                ret, frame = self.cap.read(0)
                data_init += 1
                if not ret:
                    raise Exception("Unable to read video file: {}".format(video_path))
            else:
                raise Exception("Unable to find video file: {}".format(video_path))

        if images_path != 'None':
            if os.path.exists(images_path):
                random_image_name = random.choice(os.listdir(images_path))
                cv2.imread(images_path + random_image_name)
                self.images = [cv2.imread(file) for file in sort(glob.glob(images_path + "/*.png"))]
                data_init += 1
            else:
                raise Exception("Unable to find images in: {}".format(images_path))

        if ground_truth_path != 'None':
            if os.path.exists(images_path):
                print(0)  # TODO
                self.ground_truth_path = ground_truth_path
            else:
                raise Exception("Unable to find poses in: {}".format(ground_truth_path))

        if data_init != 1:
            raise Exception("Only one data entry is required")

    def init_odometry_parameters(self):
        detector_name, descriptor_name, matcher_name, opticalFlow_name, methodForComputingE_name = get_odometry_information_from_config()
        init_parameters = 0
        norm_type_for_BFM = None
        index_params_forFLANN, search_params_forFLANN = None, dict(checks=50)

        if descriptor_name == 'ORB':
            self.descriptor = cv2.ORB_create(3000)
            norm_type_for_BFM = cv2.NORM_HAMMING
            index_params_forFLANN = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            init_parameters += 1
        elif descriptor_name == 'SIFT':
            self.descriptor = cv2.SIFT_create(3000)
            norm_type_for_BFM = cv2.NORM_HAMMING
            index_params_forFLANN = dict(algorithm=1, trees=5)
            init_parameters += 1

        if detector_name == 'ORB':
            self.detector = cv2.ORB_create(3000)
            init_parameters += 1
        elif detector_name == 'SIFT':
            self.detector = cv2.SIFT_create(3000)
            init_parameters += 1

        if matcher_name == 'BFMatcher':
            self.mather = cv2.BFMatcher(norm_type_for_BFM, crossCheck=True)
            init_parameters += 1
        elif matcher_name == 'FLANN':
            self.mather = cv2.FlannBasedMatcher(index_params_forFLANN, search_params_forFLANN)
            init_parameters += 1

        if opticalFlow_name == 'None':
            self.opticalFlow = opticalFlow_name
            init_parameters += 1

        if methodForComputingE_name == 'RANSAC':
            self.methodForComputingE = cv2.RANSAC
            init_parameters += 1
        elif methodForComputingE_name == 'LMEDS':
            self.methodForComputingE = cv2.LMEDS
            init_parameters += 1

        if init_parameters != 5:
            raise Exception(
                'One of odometry parameters not allowed {} {} {} {} {}'.format(detector_name, descriptor_name,
                                                                               matcher_name, opticalFlow_name,
                                                                               methodForComputingE_name))

    def get_frame(self, index):
        if self.images is not None:
            if len(self.images) < index:
                return self.images[index]
            else:
                return None

        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
        return frame

    def process_frame(self, image, index):
        features = self.detector.detect(image)
        descriptors = self.descriptor.compute(image, features)[1]
        if index == 0:
            self.previous_feature.init(features, descriptors)
        else:
            self.current_feature.init(features, descriptors)

    def get_matched_features(self):
        matches = self.mather.match(self.previous_feature.descriptors, self.current_feature.descriptors)

        matched_features_from_previous_image = np.float32(
            [self.previous_feature.features[m.queryIdx].pt for m in matches])
        matched_features_from_current_image = np.float32(
            [self.current_feature.features[m.trainIdx].pt for m in matches])

        return matches, matched_features_from_previous_image, matched_features_from_current_image

    def recover_pose(self, matched_features_from_previous_image, matched_features_from_current_image):
        E, mask = cv2.findEssentialMat(matched_features_from_previous_image, matched_features_from_current_image,
                                       self.camera.matrix, self.methodForComputingE)
        points, R, t, mask = cv2.recoverPose(E, matched_features_from_previous_image,
                                             matched_features_from_current_image, self.camera.matrix)
        return R, t

    def start(self):
        index = 0
        previous_image = None
        while True:
            current_image = self.get_frame(index)
            if current_image is None:
                break
            self.process_frame(current_image, index)

            if index == 0:
                index += 1
                previous_image = copy(current_image)
                continue

            matches, matched_features_from_previous_image, matched_features_from_current_image = self.get_matched_features()

            R, t = self.recover_pose(matched_features_from_previous_image, matched_features_from_current_image)
            self.position.update(R, t)

            matched_images = cv2.drawMatches(previous_image, self.previous_feature.features, current_image,
                                             self.current_feature.features, matches[:50], None,
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Matched images2", matched_images)
            cv2.waitKey(1)
            print("x: {} \n y: {} \n z: {} \n index: {}".format(self.position.get_current_x(),
                                                                self.position.get_current_y(),
                                                                self.position.get_current_z(), index))
            index += 1
            self.previous_feature = copy(self.current_feature)
            previous_image = copy(current_image)


if __name__ == '__main__':
    camera = Camera(get_camera_from_config())
    odometry = IndirectVisualOdometry()
    odometry.start()
    plt.scatter(odometry.position.x_arr, odometry.position.y_arr)
    plt.show()
    cv2.destroyAllWindows()
