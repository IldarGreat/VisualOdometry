import glob
from copy import copy

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import sort

orb = cv2.ORB_create(3000)


class Feature:
    def __init__(self):
        self.features = None
        self.descriptors = None
        pass

    def init(self, features, descriptors):
        self.features = features
        self.descriptors = descriptors
        pass


class Camera:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.matrix = self.__matrix()
        pass

    def __matrix(self):
        matrix = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fx, self.cy],
                           [0.0, 0.0, 1.0]])
        return matrix


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

    def update_position(self, previous_image, current_image):  # TODO!! Slice it
        if self.previous_feature is None:
            previous_features, previous_descriptors = orb.detectAndCompute(previous_image, None)
            self.previous_feature = Feature()
            self.previous_feature.init(previous_features, previous_descriptors)
        current_features, current_image_descriptors = orb.detectAndCompute(current_image, None)
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
        #self.position.current_rot = R.dot(self.position.current_rot)

        self.previous_feature = copy(self.current_feature)
        print("x:" + str(self.position.current_pos[0][0]))
        self.x_arr.append(self.position.current_pos[0][0])
        print("y:" + str(self.position.current_pos[2][0]))
        self.y_arr.append(self.position.current_pos[2][0])
        print("z:" + str(self.position.current_pos[1][0]))
        self.z_arr.append(self.position.current_pos[1][0])
        print()


if __name__ == '__main__':
    camera = Camera(fx=718.8560, fy=718.8560, cx=607.1928, cy=185.2157)
    odometry = IndirectVisualOdometry(camera, "data_set/images")
    index = 0
    while index < len(odometry.images) - 1:
        odometry.update_position(previous_image=odometry.images[index], current_image=odometry.images[index + 1])
        index += 1
    plt.scatter(odometry.x_arr, odometry.y_arr)
    plt.show()
    cv2.destroyAllWindows()
