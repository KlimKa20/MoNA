import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
from scipy.interpolate import CubicSpline


def integral_lsm(x_i, y_i):
    result = 0
    for i in range(0, len(x_i) - 1):
        result += (x_i[i + 1] - x_i[i]) * (y_i[i + 1] + y_i[i]) / 2
    return result


def integral_Gauss(x_i, y_i):
    spline = CubicSpline(x_i, y_i)
    result = 0
    for i in range(0, len(x_i) - 1):
        temp1 = spline(((x_i[i + 1] + x_i[i]) + (x_i[i + 1] - x_i[i]) / sqrt(3)) / 2)
        temp2 = spline(((x_i[i + 1] + x_i[i]) - (x_i[i + 1] - x_i[i]) / sqrt(3)) / 2)
        result += (x_i[i + 1] - x_i[i]) * (temp2 + temp1) / 2
    return result


def Draw(x_i, y_i):
    n = len(x_i)
    sumx, sumy = sum(x_i), sum(y_i)
    sX2 = sum([i ** 2 for i in x_i])
    a = n * sum([(x_i[i] * y_i[i]) for i in range(0, len(x_i))])
    a -= (sumx * sumy)
    a /= (n * sX2 - sumx ** 2)
    b = (sumy - a * sumx) / n
    mnk = np.array([b + a * x_i[i] for i in range(0, len(x_i))])
    plt.plot(x_i, y_i, marker='o', label="new_death")
    plt.plot(x_i, mnk, label='LSM')


def visualisation(x_i, y_i):
    Draw(x_i, y_i)
    spline = CubicSpline(x_i, y_i)
    x = np.linspace(0, 124, 1000)
    plt.plot(x, spline(x), label='spline')


if __name__ == "__main":
    data = pd.read_excel('E:/Countries-Confirmed.xlsx', header=0)
    x = np.arange(0, 124, 1)
    y = data['0204'][::-1]
    visualisation(x, y)
    lsm = integral_lsm(x, y)
    Gauss = integral_Gauss(x, y)
    print('метод наименьших квадратов:', lsm)
    print('Точность метода:', 100 - lsm / 347596, '%')
    print('метод Гаусса (с помощью сплайнов):', Gauss)
    print('Точность метода:', 100 - Gauss / 347596, '%')
    plt.legend()
    plt.show()
