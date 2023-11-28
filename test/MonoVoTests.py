import numpy as np
import pandas as pd
import yaml
import time
from src.mono_vo import IndirectVisualOdometry


def read_and_modify_one_block_of_yaml_data(filename, key, value):
    with open(f'{filename}.yaml', 'r') as f:
        data = yaml.safe_load(f)
        data['odometry'][f'{key}'] = f'{value}'
    with open(f'{filename}.yaml', 'w') as file:
        yaml.dump(data, file, sort_keys=False)


CONFIG_FILE_NAME = '../config'
POSSIBLE_DETECTORS = ['ORB', 'SIFT', 'AKAZE', 'BRISK', 'FAST', 'SURF', 'KAZE']
POSSIBLE_DESCRIPTORS = ['ORB', 'SIFT', 'AKAZE', 'BRISK', 'SURF', 'KAZE']
if __name__ == '__main__':
    pandas_detector = []
    pandas_descriptor = []
    avg_time = []
    error = []
    for detector in POSSIBLE_DETECTORS:
        for descriptor in POSSIBLE_DESCRIPTORS:
            read_and_modify_one_block_of_yaml_data('../config', 'detector', detector)
            read_and_modify_one_block_of_yaml_data('../config', 'descriptor', descriptor)
            try:
                pandas_detector.append(detector)
                pandas_descriptor.append(descriptor)
                odometry = IndirectVisualOdometry()
                start_time = time.time()
                odometry.start()
                avg_time.append(time.time() - start_time)
                #error.append(odometry.mse())
            except Exception as e:
                avg_time.append(np.NAN)
                error.append(np.NAN)
                print("Detector:" + detector + " Descriptor: " + descriptor + " Error: " + str(e))
    data = {'Detector': pandas_detector,
            'Descriptor': pandas_descriptor,
            'Avg Time:': avg_time,
            'Error:': error}
    df = pd.DataFrame(data)
    print(df)
