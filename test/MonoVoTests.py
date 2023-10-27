import yaml

from src.MonoVo import IndirectVisualOdometry


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
    for detector in POSSIBLE_DETECTORS:
        for descriptor in POSSIBLE_DESCRIPTORS:
            read_and_modify_one_block_of_yaml_data('../config', 'detector', detector)
            read_and_modify_one_block_of_yaml_data('../config', 'descriptor', descriptor)
            try:
                odometry = IndirectVisualOdometry()
                odometry.start()
            except Exception as e:
                print("Detector:" + detector + " Descriptor: " + descriptor + " Error: " + str(e))
