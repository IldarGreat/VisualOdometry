import glob
import logging
import os.path
from copy import copy

from matplotlib import pyplot as plt
from numpy import sort

from src.entity.camera import Camera
from src.entity.feature import Feature
from src.entity.position import Position
from src.utils import *


class IndirectVisualOdometry:

    def __init__(self):
        self.camera = Camera(get_camera_from_config())

        self.logging, self.see_tracking = None, None
        self.init_settings_parameters()

        self.detector, self.descriptor, self.mather, self.opticalFlow, self.methodForComputingE = init_odometry_params()

        self.cap, self.images, self.ground_truth = None, None, None
        self.init_data_parameters()

        self.position = Position()
        self.previous_feature = Feature()
        self.current_feature = Feature()

        pass

    def log(self, message):
        if self.logging is True:
            logging.info(message)

    def draw_matches(self, previous_image, current_image, matches):
        if self.see_tracking:
            matched_images = cv2.drawMatches(previous_image, self.previous_feature.features, current_image,
                                             self.current_feature.features, matches[:50], None,
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Matched set1", matched_images)
            cv2.waitKey(1)

    def init_settings_parameters(self):
        logs, see_tracking = get_settings_information_from_config()
        if logs:
            logging.basicConfig(level=logging.INFO, filename="py_log.log", filemode="w",
                                format="%(asctime)s %(levelname)s %(message)s")
            logging.getLogger().addHandler(logging.StreamHandler())
            self.logging = True
            self.log("Logging is set to True")

        if see_tracking != 'None':
            self.see_tracking = True
            self.log("See tracking is set to True")

    def init_data_parameters(self):
        camera_port, video_path, images_path, ground_truth_path, ground_truth_template = get_data_information_from_config()
        data_init = 0
        if camera_port != 'None':
            self.cap = cv2.VideoCapture(camera_port)
            data_init += 1
            self.log("Camera is detected")
            if self.cap is None or not self.cap.isOpened():
                raise Exception("Unable to open video source: {}".format(camera_port))

        if video_path != 'None':
            if os.path.exists(video_path):
                self.cap = cv2.VideoCapture(video_path)
                ret, frame = self.cap.read(0)
                data_init += 1
                self.log("Video is detected")
                if not ret:
                    raise Exception("Unable to read video file: {}".format(video_path))
            else:
                raise Exception("Unable to find video file: {}".format(video_path))

        if images_path != 'None':
            if os.path.exists(images_path):
                self.images = [cv2.imread(file) for file in sort(glob.glob(images_path + "/*.png"))]
                data_init += 1
                self.log("Image are detected")
            else:
                raise Exception("Unable to find images in: {}".format(images_path))

        if ground_truth_path != 'None':
            if os.path.exists(ground_truth_path):
                with open(ground_truth_path) as f:
                    lines = f.readlines()
                    self.ground_truth = Position()
                    for line in lines:
                        if ground_truth_template == 'KITTI':
                            position = np.array(line.split()).reshape((3, 4)).astype(np.float32)
                            self.ground_truth.append_x(position[0, 3])
                            self.ground_truth.append_y(position[1, 3])
                            self.ground_truth.append_z(position[2, 3])
            else:
                raise Exception("Unable to find poses in: {}".format(ground_truth_path))

        if data_init != 1:
            raise Exception("Only one data entry is required")

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

    def get_matched_features(self):
        matches = self.mather.match(self.previous_feature.descriptors, self.current_feature.descriptors)

        matched_features_from_previous_image = np.float32(
            [self.previous_feature.features[m.queryIdx].pt for m in matches])
        matched_features_from_current_image = np.float32(
            [self.current_feature.features[m.trainIdx].pt for m in matches])

        return matches, matched_features_from_previous_image, matched_features_from_current_image

    def recover_pose(self, matched_features_from_previous_image, matched_features_from_current_image, index):
        E, mask = cv2.findEssentialMat(matched_features_from_previous_image, matched_features_from_current_image,
                                       self.camera.matrix, self.methodForComputingE)
        points, R, t, mask = cv2.recoverPose(E, matched_features_from_previous_image,
                                             matched_features_from_current_image, self.camera.matrix)
        scale = 1
        if self.ground_truth is not None:
            px, py, pz = self.ground_truth.get_x_y_z(index - 1)
            cx, cy, cz = self.ground_truth.get_x_y_z(index)
            dif = np.power((px - cx), 2.0) + np.power((pz - cz), 2.0)
            scale = np.sqrt(dif)
        return R, t, scale

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

            R, t, scale = self.recover_pose(matched_features_from_previous_image, matched_features_from_current_image, index)
            self.position.update(R, t, scale)

            self.draw_matches(previous_image, current_image, matches)

            self.log("\nx: {}\n y: {}\n z: {}\n index: {}".format(self.position.get_current_x(),
                                                                  self.position.get_current_y(),
                                                                  self.position.get_current_z(), index))
            index += 1
            self.previous_feature = copy(self.current_feature)
            previous_image = copy(current_image)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title(
            "Detector: {},\ndescriptor: {}".format(str(self.detector), str(self.descriptor)))
        plt.plot(self.position.x_arr, self.position.z_arr, color='r', label='Estimated poses')
        plt.plot(self.ground_truth.x_arr, self.ground_truth.z_arr, color='g', label='Ground true poses')
        plt.show()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    odometry = IndirectVisualOdometry()
    odometry.start()
