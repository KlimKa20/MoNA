import matplotlib.pyplot as plt
from matplotlib import rcParams
import math as m
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

#Часть1

def CheckTriangularMatrix(matrix):
    length = len(matrix)
    for i in range(length):
        for j in range(i):
            if matrix[i][length - 1 - j] != 0:
                raise AttributeError('It is not left - triangular matrix!')


def CreatePersymmetricMatrix(matrix):
    CheckTriangularMatrix(matrix)
    length = len(matrix)
    persymmetric_matrix = np.array([np.array([complex(0)] * length)] * length)
    for i in range(length):
        for j in range(length - i):
            value = matrix[i][j]
            persymmetric_matrix[i][j] = value
            persymmetric_matrix[length - 1 - j][length - 1 - i] = value
    return persymmetric_matrix


def MakeCircles(a):
    length = len(a)
    results = []
    for i in range(length):
        results.append((a[i][i], sum([abs(a[i][j]) if i != j else 0 for j in range(length)])))
    return results


colors = {
    0: 'b',
    1: 'g',
    2: 'r',
    3: 'c',
    4: 'm',
    5: 'y',
    6: 'k',
    7: 'w'
}


def DrowCircles(circles):
    ax = plt.gca()
    for i in range(len(circles)):
        crcl = plt.Circle((circles[i][0], circles[i][0].imag), circles[i][1], color=colors[i], fill=False)
        ax.plot(circles[i][0].real, circles[i][0].imag, 'X', color=colors[i])
        ax.add_patch(crcl)
    ax.set_xlabel(u'Действительная часть')
    ax.set_ylabel(u'Комплексная часть')
    plt.title(u'Круги Гершгорина')
    plt.axis('scaled')
    plt.show()

A = np.array([[complex(0), complex(16), complex(10), complex(9), complex(13)],
              [complex(14), complex(200), complex(4), complex(2), 0],
              [complex(20), complex(14), complex(37), 0, 0],
              [complex(18), complex(11), 0, 0, 0],
              [complex(13), 0, 0, 0, 0]])
persymmetric_matrix = CreatePersymmetricMatrix(A)
circles = MakeCircles(persymmetric_matrix)
print(circles)
DrowCircles(circles)


#Часть2
#Способ 1


def AddColumn(A, B):  # Расширение матрицы А
    counter = 0
    AB = list()
    for i in A:
        AB.append(list(i))
        AB[counter].append(B[counter])
        counter += 1
    return np.array(AB)


def GaussJordanMethod(A, B):
    n = len(A)
    results = [0 for i in range(n)]
    used_rows = list()
    # Составим расширенную матрицу вида (A|B)
    AB = AddColumn(A, B)
    for k in range(n):
        scale_value, scale_index = (1, 1)
        for index in range(n):
            if index not in used_rows:
                scale_value, scale_index = AB[index][k], index
                break
        for row in range(n):
            if abs(AB[row][k] - 1) < abs(scale_value - 1) and row not in used_rows:
                scale_value, scale_index = AB[row][k], row
        used_rows.append(scale_index)
        for column in range(n + 1):
            AB[scale_index][column] /= scale_value
        for i in range(n):
            if i == scale_index:
                continue
            koeff_value = AB[i][k]
            for j in range(n + 1):
                AB[i][j] -= AB[scale_index][j] * koeff_value
    # Составляем ответ на основе полученной матрицы
    for row in range(n):
        for column in range(len(A)):
            if abs(AB[row][column]) > 0.3:
                results[column] = AB[row][n]
                break
    return results


# Конвертация вектора в характеристическое уравнение
def CreateFunctionFromArray(arr):
    l = len(arr)
    func = lambda x: x ** (l) - sum([arr[i] * x ** (l - i - 1) for i in range(l)])
    return func


def MyNewton(MyFunction, x):
    MaxError = 0.000001
    MyDerivative = lambda x: (MyFunction(x + 1e-10) - MyFunction(x)) / 1e-10
    for i in range(100):
        next_x = x - MyFunction(x) / MyDerivative(x)
        if m.fabs((next_x - x) / x) < MaxError:
            return next_x, i
        else:
            x = next_x
    else:
        raise RuntimeError('Выбрано плохое начальное приближение, достигнут лимит итераций!')


# Формирование матрицы коэффициентов A для СЛАУ
def GetMatrixFromVectors(vectors):
    matrix = [[0] * len(vectors) for i in range(len(vectors[0]))]
    l = len(vectors)
    for i in range(l):
        for j in range(len(vectors[l - i - 1])):
            matrix[j][i] = vectors[l - i - 1][j]
    return np.array(matrix)


# Формирование СЛАУ по алгоритму Крылова
def GetKrilovMatrix(A, y):
    l = len(A)
    vectors = [y]
    for i in range(l):
        vectors.append(np.dot(A, vectors[i]))
    Kr = GetMatrixFromVectors(vectors[:-1])
    return Kr, vectors[len(vectors) - 1]


# Создание вектора для домножения
def CreateY(k, n):
    a, b = k / n, k % n
    y = [0] * n
    y[b] = 1
    while (a != 0):
        y[(b + a) % n] += 1
        a -= 1
    return np.array(y)


def KrilovMethod(A):
    circles = MakeCircles(A)
    DrowCircles(circles)
    min, max = circles[0][0], circles[0][1]
    for circle in circles:
        if circle[0] - circle[1] < min:
            min = circle[0] - circle[1]
        if circle[0] + circle[1] > max:
            max = circle[0] + circle[1]
    counter = 0
    while True:
        Kr, b = GetKrilovMatrix(A, CreateY(counter, len(A)))
        if (np.linalg.det(Kr) != 0):  # Проверка, если det==0 -> используем другой вектор y (CreateY)
            counter += 1
            break
    b_array = GaussJordanMethod(Kr, b)  # Решение полученной СЛАУ для получения характеристического уравнения
    func = CreateFunctionFromArray(b_array)  # формировании lambda-выражения характеристического уравнения
    koeff = (max - min) / 100
    x = [min + koeff * i for i in range(100)]
    y = [func(val) for val in x]
    null_y = [0 for val in x]
    ax = plt.gca()
    ax.plot(x, null_y)
    ax.plot(x, y)
    plt.title(u'График функции характеристического уравнения')
    ax.set_xlabel(u'X')
    ax.set_ylabel(u'Y')
    plt.show()
    newton_min = MyNewton(func, min)[0]  # нахождение максимального и минимального значений методом Ньютона
    newton_max = MyNewton(func, max)[0]
    return newton_min, newton_max


def CountAndSHowMinMax(A):
    res_A = KrilovMethod(A)
    print("Результаты вычислений:\n Min:\t{}\t Max: {}\n".format(res_A[0], res_A[1]))
    st_A = np.linalg.eig(A)
    print("Результаты вычислений стандартной функцией:\n Min:\t{}\t Max: {}\n".format(min(st_A[0]), max(st_A[0])))


B = np.array([
    [2.2, 1, 0.5, 2],
    [1, 1.3, 2, 1],
    [0.5, 2, 0.5, 1.6],
    [2, 1, 1.6, 2]
])
print("Не-персимметрическая матрица:")
CountAndSHowMinMax(B)
print("Персимметрическая матрица из прошлого задания:")
CountAndSHowMinMax(persymmetric_matrix)


#способ2

def magnitude(v):
    return m.sqrt(sum(vi ** 2 for vi in v))

def scalar_multiply(c, v):
    return sum([c[i] * v[i] for i in range(len(v))])

def DegreeMethod(A):
    r_k = np.array([random.randint(-100, 100)] * len(A))
    A_rk = np.dot(A,r_k)
    last_err, cur_err = 2,1
    nu_k, nu_k1 = scalar_multiply(r_k, A_rk)/scalar_multiply(r_k, r_k), 100000
    while abs(last_err-cur_err)/last_err>0.0000001:
        #print(abs(nu_k - nu_k1)/nu_k)
        #print("REL: {}".format(abs(last_err-cur_err)/last_err))
        A_rk = np.dot(A,r_k)
        r_k = np.dot(A,r_k)/magnitude(A_rk)
        nu_k1 = scalar_multiply(r_k, A_rk)/scalar_multiply(r_k, r_k)
        last_err, cur_err = cur_err, abs(nu_k - nu_k1)/nu_k
    return nu_k1

def CountAndSHowMinMax(A):
    inv_A = np.linalg.inv(A)
    res_A = 1/DegreeMethod(inv_A), DegreeMethod(A)
    print("Результаты вычислений:\n Min:\t{}\t Max: {}\n".format(res_A[0],res_A[1]))
    st_A = np.linalg.eig(A)
    print(st_A[0])
    print("Результаты вычислений стандартной функцией:\n Min:\t{}\t Max: {}\n".format(min(st_A[0]), max(st_A[0])))

B = np.array([
    [2.2, 1, 0.5, 2],
    [1, 1.3, 2, 1],
    [0.5, 2, 0.5, 1.6],
    [2, 1, 1.6, 2]
])
print("Не-персимметрическая матрица:")
CountAndSHowMinMax(B)
print("Персимметрическая матрица из прошлого задания:")
CountAndSHowMinMax(persymmetric_matrix)

#Часть3

print('Для матрицы A:')
C = np.array([[complex(2), complex(-1), complex(0), complex(0)],
             [complex(-1), complex(2), complex(-1), complex(0)],
             [complex(0), complex(-1), complex(0), complex(0)],
             [complex(0), complex(0), complex(0), complex(0)]])
per_c = CreatePersymmetricMatrix(C)
circles = MakeCircles(per_c)
DrowCircles(circles)
I4_matrix = np.array([
    [4,0,0,0],
    [0,4,0,0],
    [0,0,4,0],
    [0,0,0,4]])
CI4 = per_c - I4_matrix
circles1 = MakeCircles(CI4)
print('Для матрицы A-4I:')
DrowCircles(circles1)
