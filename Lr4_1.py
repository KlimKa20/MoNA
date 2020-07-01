import matplotlib.pyplot as plt
from matplotlib import rcParams
import math as m
import sympy as sym
import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import CubicSpline


def GraphInterpretation(X: list, Y: list, f: list, names: list):
    rborder = 1
    rcParams['figure.subplot.right'] = rborder
    rcParams['figure.subplot.hspace'] = 0.40
    rcParams['figure.subplot.wspace'] = 0.40

    fig = plt.figure()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 7))

    for i, ax in enumerate(fig.axes):
        minimum = min(X[i])
        maximum = max(X[i])
        koeff = (maximum - minimum) / 100
        x = [minimum + koeff * i for i in range(100)]
        ax.plot(x, [f[i](j) for j in x], label='Аппроксимация', c='orange')
        ax.scatter(X[i], Y[i], label='Значения функции', s=15, c='brown')
        ax.title.set_text(names[i])
        ax.grid(True)
        ax.set_xlabel(u'X')
        ax.set_ylabel(u'Y')
        ax.legend(loc='best', frameon=False)  # легенда для области рисования ax
    plt.show()

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


def LeastSquareMethod(X: list, Y: list, N: int):
    S, B = CreateMatrix(X, Y, N, 1)
    A = np.linalg.inv(S).dot(B)
    func = lambda x: float(A[0] + A[1] * float(x))
    return func


def PolynomApproximation(X: list, Y: list, N: int, degree: int):
    if degree <= 0 or degree > 10:
        raise AttributeError('The degree of the polynomial must be within (1;10). Not {}.'.format(B))
    S, B = CreateMatrix(X, Y, N, degree)
    A = np.linalg.inv(S).dot(B)
    func = lambda x: A[0] + sum(A[i] * x ** i for i in range(1, degree + 1))
    return func


class Spline:
    def __init__(self, a, b, c, d, x):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x = x


def build_spline(x, y, n):
    splines = [Spline(0, 0, 0, 0, 0) for _ in range(0, n)]
    for i in range(0, n):
        splines[i].x = x[i]
        splines[i].a = y[i]
    alpha = [0.0 for _ in range(0, n - 1)]
    beta = [0.0 for _ in range(0, n - 1)]
    splines[0].c = splines[n - 1].c = 0.0
    for i in range(1, n - 1):
        h = x[i] - x[i - 1]
        h1 = x[i + 1] - x[i]
        A = h
        C = 2.0 * (h + h1)
        B = h1
        F = 6.0 * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h)
        z = (A * alpha[i - 1] + C)
        alpha[i] = -B / z
        beta[i] = (F - A * beta[i - 1]) / z

    for i in range(n - 2, 0, -1):
        splines[i].c = alpha[i] * splines[i + 1].c + beta[i]

    for i in range(n - 1, 0, -1):
        h = x[i] - x[i - 1]
        splines[i].d = (splines[i].c - splines[i - 1].c) / h
        splines[i].b = h * (2.0 * splines[i].c + splines[i - 1].c) / 6.0 + (y[i] - y[i - 1]) / h
    func = lambda x: interpolate(splines, x)
    return func


def interpolate(splines, x):
    if not splines:
        return None
    n = len(splines)
    s = Spline(0, 0, 0, 0, 0)

    if x <= splines[0].x:
        s = splines[0]
    elif x >= splines[n - 1].x:
        s = splines[n - 1]
    else:
        i = 0
        j = n - 1
        while i + 1 < j:
            k = i + (j - i) // 2
            if x <= splines[k].x:
                j = k
            else:
                i = k
        s = splines[j]

    dx = x - s.x
    return s.a + (s.b + (s.c / 2.0 + s.d * dx / 6.0) * dx) * dx


def CountAndPlotApproximations(X, Y,test,func, degree):
    name = ['e^(2x)*cos(3x)', 'sin(ln(x))', 'ln(x)', 'cos(x) + sin(x)']
    LeastSqare_res = [LeastSquareMethod(X[0], Y[0], len(X[0])), LeastSquareMethod(X[1], Y[1], len(X[1])),
                      LeastSquareMethod(X[2], Y[2], len(X[2])), LeastSquareMethod(X[3], Y[3], len(X[3]))]
    GraphInterpretation(X, Y, LeastSqare_res, name)
    print("None")
    for i in range(0,4):
        print("\nФункция:{0}".format(name[i]))
        for j in range(0, len(test)):
            print("S({0}) = {1}".format(test[j],LeastSqare_res[i](test[j])))
            print("delta({0}) = {1}".format(test[j], abs(LeastSqare_res[i](test[j]) - func[i](test[j]))))
    PolynomApproximation_res = [PolynomApproximation(X[0], Y[0], len(X[0]), degree),
                                PolynomApproximation(X[1], Y[1], len(X[1]), degree),
                                PolynomApproximation(X[2], Y[2], len(X[2]), degree),
                                PolynomApproximation(X[3], Y[3], len(X[3]), degree)]
    print("None")
    for i in range(0, 4):
        print("\nФункция:{0}".format(name[i]))
        for j in range(0, len(test)):
            print("S({0}) = {1}".format(test[j], PolynomApproximation_res[i](test[j])))
            print("delta({0}) = {1}".format(test[j], abs(PolynomApproximation_res[i](test[j]) - func[i](test[j]))))


    GraphInterpretation(X, Y, PolynomApproximation_res, name)

    Square = [build_spline(X[0], Y[0], len(X[0])),
                             build_spline(X[1], Y[1], len(X[1])),
                             build_spline(X[2], Y[2], len(X[2])),
                             build_spline(X[3], Y[3], len(X[3]))]
    print("Кубические сплайны")
    for i in range(0, 4):
        print("\nФункция:{0}".format(name[i]))
        for j in range(0, len(test)):
            print("S({0}) = {1}".format(test[j], Square[i](test[j])))
            print("delta({0}) = {1}".format(test[j], abs(Square[i](test[j]) - func[i](test[j]))))

    GraphInterpretation(X, Y, Square, name)


f_1 = lambda x: m.exp(2 * x) * m.cos(3 * x)
f_2 = lambda x: m.sin(m.log(x))
f_3 = lambda x: m.log(x)
f_4 = lambda x: m.cos(x) + m.sin(x)
test = [0.25,0.51,0.99,1.09,1.89,2.39]
X_values = [[0, 0.3, 0.6], [2.0, 2.4, 2.6], [1.0, 1.1, 1.3, 1.4], [0.0, 0.25, 0.5, 1.0]]
Y_values = [list(f_1(x) for x in X_values[0]), list(f_2(x) for x in X_values[1]),
            list(f_3(x) for x in X_values[2]), list(f_4(x) for x in X_values[3])]
CountAndPlotApproximations(X_values, Y_values, test,[f_1,f_2,f_3,f_4] ,2)
