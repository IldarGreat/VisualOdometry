import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity


def load_gt(path):
    with open(path) as f:
        lines = f.readlines()
        x_gt = []
        y_gt = []
        z_gt = []
        for line in lines:
            position = np.array(line.split()).reshape((3, 4)).astype(np.float32)
            x_gt.append(position[0, 3])
            y_gt.append(position[1, 3])
            z_gt.append(position[2, 3])
    return x_gt, y_gt, z_gt


def load_dso(path, multiplying_scale, addition_scale):
    dso_data = multiplying_scale * np.loadtxt(path)
    x_dso = dso_data[:, 1]
    y_dso = dso_data[:, 2]
    z_dso = dso_data[:, 3] + addition_scale
    return x_dso, y_dso, z_dso


def load_ldso(path, multiplying_scale, addition_scale, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    ldso_data = multiplying_scale * np.loadtxt(path)
    x_dso = ldso_data[:, 4]
    y_dso = ldso_data[:, 8]
    z_dso = ldso_data[:, 12] + addition_scale
    rotated_points = np.dot(rotation_matrix, np.vstack((x_dso, y_dso)))
    x_dso = rotated_points[0, :]
    y_dso = rotated_points[1, :]
    return x_dso, y_dso, z_dso


def load_indirect(path):
    vo_data = np.loadtxt(path)
    x_vo = vo_data[:, 1]
    y_vo = vo_data[:, 2]
    z_vo = vo_data[:, 3]
    return x_vo, y_vo, z_vo


def load_ptam(path):
    ptam_data = np.loadtxt(path)
    x_ptam = ptam_data[:, 0]
    y_ptam = ptam_data[:, 1]
    z_ptam = ptam_data[:, 2]
    return x_ptam, y_ptam, z_ptam


def metrics(x1, z1, x2, z2):
    model1 = np.hypot(x1, z1)
    model2 = np.hypot(x2, z2)
    mse = mean_squared_error(model1, model2) / len(model1)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(model1, model2) / len(model1)
    r2 = r2_score(model1, model2) / len(model1)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae)
    print("R-squared (Coefficient of Determination):", r2)


def plot_with_gt(x_gt, z_gt, x, z, title):
    if len(x) < len(x_gt):
        x_gt = signal.resample(x_gt, len(x))
        z_gt = signal.resample(z_gt, len(z))
    elif len(x) > len(x_gt):
        x = signal.resample(x, len(x_gt))
        z = signal.resample(z, len(z_gt))
    print("----------------" + title + "----------------")
    metrics(x_gt, z_gt, x, z)
    print("-----------------------------------")
    plt.plot(x_gt, z_gt, label='gt')
    plt.plot(x, z, label=title)
    plt.xlabel('x (м)')
    plt.ylabel('z (м)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    path = 'ldso/ldso7.txt'
    gt_path = 'gt/gt7.txt'
    x, y, z = load_ldso(path, 18, 150, 0)
    # x, y, z = load_ptam(path)
    x_gt, y_gt, z_gt = load_gt(gt_path)
    plot_with_gt(x_gt, z_gt, x, z, 'ldso')
