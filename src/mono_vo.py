import logging
from copy import copy

from matplotlib import pyplot as plt

from entity.camera import Camera
from entity.feature import Feature
from utils.compute_utils import compute_error
from utils.config_utils import *

lk_params = dict(winSize=(21, 21),
                 criteria=(cv2.TERM_CRITERIA_EPS |
                           cv2.TERM_CRITERIA_COUNT, 30, 0.03))


class IndirectVisualOdometry:

    def __init__(self):
        logging.basicConfig(level=logging.INFO, filename="IndirectVisualOdometry.log", filemode="w",
                            format="%(asctime)s %(levelname)s %(message)s")
        logging.getLogger().addHandler(logging.StreamHandler())

        self.logging = True
        self.camera = Camera(get_camera_from_config())

        self.see_tracking = init_settings_parameters()

        self.detector, self.descriptor, self.mather, self.opticalFlow, self.methodForComputingE = init_odometry_params()

        self.cap, self.images, self.ground_truth = init_data_parameters()

        self.position = Position()
        self.previous_feature = Feature()
        self.current_feature = Feature()

        pass

    def get_frame(self, index):
        if self.images is not None:
            if len(self.images) > index:
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

    def get_matched_features(self, index):
        #
        matches = self.mather.knnMatch(self.previous_feature.descriptors, self.current_feature.descriptors, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)

        kp1 = np.float32([self.previous_feature.features[m.queryIdx].pt for m in good])
        kp2 = np.float32([self.current_feature.features[m.trainIdx].pt for m in good])

        return matches, kp1, kp2
        # return kp1, kp2

    def recover_pose(self, matched_features_from_previous_image, matched_features_from_current_image, index):
        E, mask = cv2.findEssentialMat(matched_features_from_previous_image, matched_features_from_current_image,
                                       self.camera.matrix, self.methodForComputingE)
        points, R, t, mask = cv2.recoverPose(E, matched_features_from_previous_image,
                                             matched_features_from_current_image, self.camera.matrix)
        scale = 1
        if self.ground_truth is not None and index <= len(self.ground_truth.x_arr) - 1:
            x_prev, y_prev, z_prev = self.ground_truth.get_x_y_z(index - 1)
            x_cur, y_cur, z_cur = self.ground_truth.get_x_y_z(index)
            dif = np.power((x_cur - x_prev), 2.0) + np.power((z_cur - z_prev), 2.0) + np.power((y_cur - y_prev), 2.0)
            scale = np.sqrt(dif)
        return R, t, scale

    def start(self):
        index = 0
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2)
        # previous_image = None
        with open('../test/matching_results/indirect/result1.txt', 'w') as f:
            while True:
                current_image = self.get_frame(index)
                if current_image is None:
                    break
                self.process_frame(current_image, index)

                if index == 0:
                    index += 1
                    # previous_image = copy(current_image)
                    continue

                matches, matched_features_from_previous_image, matched_features_from_current_image = self.get_matched_features(
                    index)

                R, t, scale = self.recover_pose(matched_features_from_previous_image,
                                                matched_features_from_current_image,
                                                index)
                self.position.update(R, t, scale)
                ax1.plot(self.position.get_current_x(), self.position.get_current_z(), 'ro')
                ax1.plot(self.ground_truth.get_x_y_z(index-1)[0], self.ground_truth.get_x_y_z(index-1)[2], 'bo')
                ax2.plot(index,
                         compute_error(self.position.get_x_y_z(index - 1), self.ground_truth.get_x_y_z(index - 1),
                                       'MSE'), 'ro')
                plt.draw()
                plt.pause(0.01)

                index += 1
                self.previous_feature = copy(self.current_feature)
                # previous_image = copy(current_image)
                data = str(
                    index) + ' ' + str(self.position.get_current_x()) + ' ' + str(
                    self.position.get_current_y()) + ' ' + str(self.position.get_current_z())
                print(data)
                f.write(data + '\n')


if __name__ == '__main__':
    odometry = IndirectVisualOdometry()
    odometry.start()
