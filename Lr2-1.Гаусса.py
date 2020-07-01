import numpy
import math


def check_diagonal(a, n):
    for i in range(n):
        if a[i][j] == 0:
            for j in range(n):
                if j == i:
                    continue
                if a[j][i] != 0 and a[i][j] != 0:
                    for k in range(k):
                        a[j][k], a[i][k] = a[i][k], a[j][k]
                    b[j], b[i] = b[i], b[j]


def gauss(a, b, n):
    x = {}
    k = 0
    eps = 0.00001
    while k < n:
        max = math.fabs(a[k][k])
        index = k
        for i in range(k + 1, n):
            if math.fabs(a[i][k]) > max:
                max = abs(a[i][k])
                index = i
        if max < eps:
            raise Exception("Нулевой столбец")
        for j in range(n):
            a[k][j], a[index][j] = a[index][j], a[k][j]
            b[k], b[index] = b[index], b[k]
        for i in range(k, n):
            temp = a[i][k]
            if math.fabs(temp) < eps:
                continue
            for j in range(n):
                a[i][j] = a[i][j] / temp
            b[i] = b[i] / temp
            if i == k:
                continue
            for j in range(n):
                a[i][j] = a[i][j] - a[k][j]
            b[i] = b[i] - b[k]
        k += 1
    for k in range(n - 1, -1, -1):
        x[k] = b[k]
        for i in range(k):
            b[i] = b[i] - a[i][k] * x[k]
    return list(map(float, x.values()))


a = numpy.array([[3.389, 0.273, 0.126, 0.418],
                 [0.329, 2.796, 0.179, 0.287],
                 [0.186, 0.275, 2.987, 0.316],
                 [0.197, 0.219, 0.274, 3.127]])
b = numpy.array([0.144, 0.297, 0.529, 0.869])
x = gauss(a, b, 4)
x.reverse()
print(x)
