import numpy as np
import matplotlib.pyplot as plt

# Ваши исходные данные
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Угол поворота в радианах
angle = np.pi

# Матрица поворота
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

# Применение поворота к массиву точек
rotated_points = np.dot(rotation_matrix, np.vstack((x, y)))

# Извлечение повернутых координат
rotated_x = rotated_points[0, :]
rotated_y = rotated_points[1, :]

# Построение повернутого графика
plt.plot(rotated_x, rotated_y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Повернутый график')
plt.show()