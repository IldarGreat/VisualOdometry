import glob
from copy import copy

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import sort

from src.entity.camera import Camera
from src.entity.feature import Feature
from src.entity.position import Position
from src.utils import get_camera_from_config

orb = cv2.ORB_create(3000)
sift = cv2.SIFT_create()


class IndirectVisualOdometry:

    def __init__(self, camera, path, draw_result=True, detector="ORB", descriptor="BRIEF", matcher="Brute-Force",
                 quite=True):
        self.camera = camera
        self.path = path
        self.detector = detector
        self.descriptor = descriptor
        self.images = [cv2.imread(file) for file in sort(glob.glob(path + "/*.png"))]
        self.position = Position()
        self.draw_result = draw_result
        self.previous_feature = None
        self.current_feature = Feature()
        self.quite = quite
        self.matcher = matcher
        self.x_arr = []
        self.y_arr = []
        self.z_arr = []
        pass

    @staticmethod
    def detect_features(detector, image):
        if detector == 'ORB':
            return orb.detect(image)
        elif detector == 'SIFT':
            return sift.detect(image)

    @staticmethod
    def compute_descriptor(descriptor, image, features):
        if descriptor == 'ORB':
            return orb.compute(image, features)[1]
        elif descriptor == 'SIFT':
            return sift.compute(image, features)

    def process_first_frame(self, detector, descriptor, image):
        first_features = IndirectVisualOdometry.detect_features(detector, image)
        first_descriptors = IndirectVisualOdometry.compute_descriptor(descriptor, image, first_features)
        self.previous_feature = Feature()
        self.previous_feature.init(first_features, first_descriptors)

    def start(self):
        index = 0
        while index < len(odometry.images) - 1:
            previous_image = self.images[index]
            current_image = self.images[index + 1]
            if index == 0:
                self.process_first_frame('ORB', 'ORB', self.images[0])
            current_features = self.detect_features('ORB', current_image)
            current_image_descriptors = self.compute_descriptor('ORB', current_image, current_features)
            self.current_feature.init(current_features, current_image_descriptors)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # BruteForce Matcher
            matches = bf.match(self.previous_feature.descriptors, self.current_feature.descriptors)

            matched_features_from_previous_image = np.float32(
                [self.previous_feature.features[m.queryIdx].pt for m in matches])
            matched_features_from_current_image = np.float32(
                [self.current_feature.features[m.trainIdx].pt for m in matches])

            if self.draw_result:
                matched_images = cv2.drawMatches(previous_image, self.previous_feature.features, current_image,
                                                 self.current_feature.features, matches[:50], None,
                                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow("Matched images2", matched_images)
                cv2.waitKey(1)

            E, mask = cv2.findEssentialMat(matched_features_from_previous_image, matched_features_from_current_image,
                                           self.camera.matrix, cv2.RANSAC)
            points, R, t, mask = cv2.recoverPose(E, matched_features_from_previous_image,
                                                 matched_features_from_current_image, self.camera.matrix)

            self.position.current_pos += self.position.current_rot.dot(t)
            self.position.current_rot = R.dot(self.position.current_rot)
            # self.position.current_rot = R.dot(self.position.current_rot)

            self.previous_feature = copy(self.current_feature)
            print("x:" + str(self.position.current_pos[0][0]))
            self.x_arr.append(self.position.current_pos[0][0])
            print("y:" + str(self.position.current_pos[2][0]))
            self.y_arr.append(self.position.current_pos[2][0])
            print("z:" + str(self.position.current_pos[1][0]))
            self.z_arr.append(self.position.current_pos[1][0])
            print()
            index += 1


if __name__ == '__main__':
    camera = Camera(get_camera_from_config())
    odometry = IndirectVisualOdometry(camera, "../data_set/images")
    odometry.start()
    plt.scatter(odometry.x_arr, odometry.y_arr)
    plt.show()
    cv2.destroyAllWindows()
