import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# Часть 1

# Содание матрицы
def create_matrix(m, alpha):
    list_matrix = list()
    for i in range(0, m):
        temper_vector = list()
        for j in range(0, m):
            if i == j:
                temper_vector.append(1 + 2 * alpha)
            elif abs(j - i) == 1:
                temper_vector.append(-alpha)
            else:
                temper_vector.append(0)
        list_matrix.append(temper_vector)
    return np.array(list_matrix)


def smid(A):
    B = []
    B.append(A[:, 0])
    for i in range(1, len(A[0, :])):
        b0 = sum([(sum(A[:, i] * B[x]) / sum(B[x] * B[x]) * B[x]) for x in range(0, i)])
        b = A[:, i] - b0
        B.append(b)
    for i in range(0, len(A[0, :])):
        X = math.sqrt(sum(x ** 2 for x in B[i]))
        B[i] = B[i] / X
    B = np.array(B).T
    return B


def QRmethod(A, errors):
    curent_different = 1
    Q = np.zeros((len(A[:, 0]), len(A[:, 0])))
    while (curent_different > errors):
        Q_new = smid(A)
        R = np.dot(Q_new.T, A)
        A1 = np.dot(R, Q_new)
        curent_different = abs(np.linalg.det(Q - Q_new))
        Q, A = Q_new, A1
    return [A[x, x] for x in range(0, len(A[0, :]))]


koeff = 5
alpha = 0.8
A = create_matrix(koeff - 1, alpha)
print('Исходная матрица\n{}'.format(A))

v = QRmethod(A, 0.0000000001)
print("собственные значения, полученные методом QR-разложения:\n{}".format(v))

vector = [1 + 4 * alpha * (np.sin(np.pi * x / 2 / koeff)) ** 2 for x in range(1, koeff)]
vector.reverse()
print("Правильные собственные значения из условия:\n{}".format(vector))

# Часть 2

bb = []
for i in range(0, int(len(v) / 2)):
    vv = []
    vv += [0] * i
    vv += (v[x] for x in range(1, len(v), 2))
    bb.append(vv + [0] * (1 - i if len(v) % 2 == 0 else 2 - i))
    vv = []
    vv += [0] * i
    vv += (v[x] for x in range(0, len(v), 2))
    bb.append(vv + [0] * (1 - i if len(v) % 2 == 0 else 1 - i))
bb = np.array(bb)
for i in range(0, len(v) - 1):
    A1 = np.delete(bb, [i for i in range(len(v) - 1, i, -1)], axis=1)
    A1 = np.delete(A1, [i for i in range(len(v) - 1, i, -1)], axis=0)
    if np.linalg.det(A1) < 0:
        break
print("Область устойчивости матрицы {0}".format(i))
