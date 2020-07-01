import matplotlib.pyplot as plt
from matplotlib import rcParams
import math as m
import sympy as sym
import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import CubicSpline


def GetDataForAnalys(data: pd.core.frame.DataFrame):
    daily_sum_list = [m.log(65)]
    sum_prev = sum(data['0122'])
    headers = list(data)
    for i in headers[2:]:
        sum_curr = sum(data[i])
        daily_sum_list.append(m.log(sum_curr - sum_prev))
        sum_prev = sum_curr
    return daily_sum_list


colors = {0: 'brown', 1: 'red', 2: 'green', 3: 'blue', 4: 'orange'}


def PolynomApproximation(X: list, Y: list, N: int, degree: int):
    if degree <= 0 or degree > 10:
        raise AttributeError('The degree of the polynomial must be within (1;10). Not {}.'.format(B))
    S, B = CreateMatrix(X, Y, N, degree)
    A = np.linalg.inv(S).dot(B)
    func = lambda x: A[0] + sum(A[i] * x ** i for i in range(1, degree + 1))
    return func


def CreateMatrix(X: list, Y: list, N: int, degree: int):
    S = [[0.] * (degree + 1) for i in range(degree + 1)]
    S[0][0] = N
    for i in range(1, 2 * (degree) + 1):
        value = sum(x ** i for x in X)
        for j in range(i + 1):
            if j < degree + 1 and i - j < degree + 1 and i - j >= 0:
                S[i - j][j] = value
    B = [0] * (degree + 1)
    for i in range(degree + 1):
        B[i] = sum(Y[j] * X[j] ** i for j in range(N))
    # print('S:\n{}\n\nB:\n{}0'.format(S,B))
    return np.array(S), np.array(B)


def DisplayCoronaResults(X: list, Y: list, titles: list):
    rborder = 1
    rcParams['figure.subplot.right'] = rborder
    rcParams['figure.subplot.hspace'] = 0.40
    rcParams['figure.subplot.wspace'] = 0.40

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 20))

    for i, ax in enumerate(fig.axes):
        if i == 0:
            ax.plot(X, Y[0], c='brown')
        else:
            ax.scatter(X, Y[0], s=15, c=colors[0], label=titles[0])
            ax.plot(X, Y[i], c=colors[i], label=titles[i])
        ax.grid(True)
        ax.set_xlabel(u'0.1х = 1 день')
        ax.set_ylabel(u'logn(число заболевших)')
        ax.legend(loc='best', frameon=True)
    plt.show()
    spline = CubicSpline(X, Y[0])
    x = np.linspace(0.1, 10, 1000)
    plt.plot(x, spline(x), color='#58b970', linewidth=2)
    plt.show()


data_for_analys = pd.read_excel('D:/Countries-Confirmed.xlsx', header=0)
daily_sum_list = GetDataForAnalys(data_for_analys)
X = list(range(1, len(daily_sum_list) + 1))
X = [(x * 1.0) / 10 for x in X]

Y = [daily_sum_list]
for i in range(1, 8, 2):
    f = PolynomApproximation(X, daily_sum_list, len(daily_sum_list), i)
    Y.append([f(x) for x in X])

DisplayCoronaResults(X, Y, ['current', 'linear', 'hyperbolic', '5-degree', '7-degree'])
