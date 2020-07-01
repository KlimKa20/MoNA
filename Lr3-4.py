import matplotlib.pyplot as plt
from matplotlib import rcParams
import math as m
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')


def GetTwoLists(l):
    x = []
    y = []
    for i in l:
        x.append(i[0])
        y.append(i[1])
    return x, y


def GraphInterpretation(nations_dict: dict, gender_dict: dict, age_dict_6_12, age_dict_13_18, age_dict_19_24,
                        age_dict_25_30):
    rborder = 1
    rcParams['figure.subplot.right'] = rborder
    rcParams['figure.subplot.hspace'] = 0.40
    rcParams['figure.subplot.wspace'] = 0.40

    fig = plt.figure()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 7))

    for i, ax in enumerate(fig.axes):
        if i == 0:
            for item in gender_dict.items():
                ax.scatter(*GetTwoLists(item[1][0]), label=item[0], s=1.5, c=item[1][1])
                ax.title.set_text('Зависимость от пола')
        elif i == 1:
            for item in nations_dict.items():
                ax.title.set_text('Зависимость от национальности')
                ax.scatter(*GetTwoLists(item[1][0]), label=item[0], s=1.5, c=item[1][1])
        ax.grid(True)
        ax.set_xlabel(u'Компонента 2')
        ax.set_ylabel(u'Компонента 1')
        ax.legend(loc='best', frameon=False)  # легенда для области рисования ax
    plt.show()

    fig_age = plt.figure()
    fig_age, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 7))

    for i, ax in enumerate(fig_age.axes):
        if i == 0:
            for item in age_dict_6_12.items():
                ax.scatter(*GetTwoLists(item[1][0]), label=item[0], s=2, c=item[1][1])
                ax.title.set_text('Возраст 6-12 лет')
        elif i == 1:
            for item in age_dict_13_18.items():
                ax.scatter(*GetTwoLists(item[1][0]), label=item[0], s=2, c=item[1][1])
                ax.title.set_text('Возраст 13-18 лет')
        elif i == 2:
            for item in age_dict_19_24.items():
                ax.scatter(*GetTwoLists(item[1][0]), label=item[0], s=2, c=item[1][1])
                ax.title.set_text('Возраст 19-24 года')
        elif i == 3:
            for item in age_dict_25_30.items():
                ax.scatter(*GetTwoLists(item[1][0]), label=item[0], s=2, c=item[1][1])
                ax.title.set_text('Возраст 25-30 лет')

        ax.grid(True)
        ax.set_xlabel(u'Компонента 2')
        ax.set_ylabel(u'Компонента 1')
        ax.legend(loc='best', frameon=False)  # легенда для области рисования ax
    plt.show()


def GraphBarsInterpretation(age_values, nations_values, gender_values):
    rborder = 1
    rcParams['figure.subplot.right'] = rborder
    rcParams['figure.subplot.hspace'] = 0.40
    rcParams['figure.subplot.wspace'] = 0.40

    fig = plt.figure()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    for i, ax in enumerate(fig.axes):
        if i == 0:
            for item in gender_values.items():
                ax.barh(item[0], item[1][0][0], color=item[1][1], height=0.9, alpha=0.7)
                ax.title.set_text('Зависимость от пола')
        elif i == 1:
            for item in nations_values.items():
                ax.barh(item[0], item[1][0][0], color=item[1][1], height=0.9, alpha=0.7)
                ax.title.set_text('Зависимость от национальности')
        ax.grid(True)
        ax.set_ylabel(u'Ср. арифм. относительно M[x]')
        ax.legend(loc='best', frameon=False)  # легенда для области рисования ax
    plt.show()

    fig_age = plt.figure()
    fig_age, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    for i, ax in enumerate(fig_age.axes):
        for item in age_values.items():
            ax.bar(item[0], item[1][0][0], color=item[1][1])
            ax.title.set_text('Зависимость от возраста')

        ax.grid(True)
        ax.set_ylabel(u'Среднее арифметическое относительно среднего')
        ax.legend(loc='best', frameon=False)  # легенда для области рисования ax
    plt.show()


def DrawGraphics(data, results):
    age_values, nations_values, gender_values = dict(), dict(), dict()
    nations_dict, gender_dict, age_dict_6_12, age_dict_13_18, age_dict_19_24, age_dict_25_30 = dict(), dict(), dict(), dict(), dict(), dict()
    for i in range(len(data)):
        if data['Y2'][i] not in nations_dict.keys():
            nations_dict[data['Y2'][i]] = ([], "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
        if data['Y3'][i] not in gender_dict.keys():
            gender_dict[data['Y3'][i]] = ([], "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
        age = int(data['Y4'][i])
        if age <= 12:
            if data['Y4'][i] not in age_dict_6_12.keys():
                age_dict_6_12[data['Y4'][i]] = (
                [], "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
            age_dict_6_12[data['Y4'][i]][0].append((results[0][i], results[1][i]))
        elif age <= 18:
            if data['Y4'][i] not in age_dict_13_18.keys():
                age_dict_13_18[data['Y4'][i]] = (
                [], "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
            age_dict_13_18[data['Y4'][i]][0].append((results[0][i], results[1][i]))
        elif age <= 24:
            if data['Y4'][i] not in age_dict_19_24.keys():
                age_dict_19_24[data['Y4'][i]] = (
                [], "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
            age_dict_19_24[data['Y4'][i]][0].append((results[0][i], results[1][i]))
        else:
            if data['Y4'][i] not in age_dict_25_30.keys():
                age_dict_25_30[data['Y4'][i]] = (
                [], "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
            age_dict_25_30[data['Y4'][i]][0].append((results[0][i], results[1][i]))

        if data['Y2'][i] not in nations_values.keys():
            nations_values[data['Y2'][i]] = (
            [0, 0], "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
        if data['Y3'][i] not in gender_values.keys():
            gender_values[data['Y3'][i]] = (
            [0, 0], "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
        if data['Y2'][i] not in age_values.keys():
            age_values[data['Y4'][i]] = ([0, 0], "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]))

        age_values[data['Y4'][i]][0][0] += results[0][i]
        age_values[data['Y4'][i]][0][1] += 1
        nations_values[data['Y2'][i]][0][0] += results[0][i]
        nations_values[data['Y2'][i]][0][1] += 1
        gender_values[data['Y3'][i]][0][0] += results[0][i]
        gender_values[data['Y3'][i]][0][1] += 1
        nations_dict[data['Y2'][i]][0].append((results[0][i], results[1][i]))
        gender_dict[data['Y3'][i]][0].append((results[0][i], results[1][i]))

    for key in age_values.keys():
        age_values[key][0][0] /= age_values[key][0][1]
    for key in gender_values.keys():
        gender_values[key][0][0] /= gender_values[key][0][1]
    for key in nations_values.keys():
        nations_values[key][0][0] /= nations_values[key][0][1]
    GraphBarsInterpretation(age_values, nations_values, gender_values)
    GraphInterpretation(nations_dict, gender_dict, age_dict_6_12, age_dict_13_18, age_dict_19_24, age_dict_25_30)


# Формирование матрицы H для метода Якоби
def GetMatrixH(i, j, phi, n):
    H = [[0.] * n for i in range(n)]
    for k in range(n):
        H[k][k] = 1
    H[j][j] = H[i][i] = m.cos(phi)
    H[i][j] = -m.sin(phi)
    H[j][i] = m.sin(phi)
    return np.array([np.array(item) for item in H])


# Получение максимального по модулю значения из наддиагонильных элементов
def GetMaxValue(A):
    max, c_i, c_j = 0, -1, -1
    for i in range(len(A)):
        for j in range(len(A) - i - 1):
            if abs(A[i][len(A) - j - 1]) > abs(max):
                max, c_i, c_j = A[i][len(A) - j - 1], i, len(A) - j - 1
    return max, c_i, c_j


# Метод Якоби для нахождения собственных значений и собственных векторов
def YacobiMethod(A, err):
    eig_is_changed = False
    eigVectMatrix = np.array([[0.] * len(A)] * len(A))
    Aij_max, i, j = GetMaxValue(A)
    counter = 0
    while abs(Aij_max) > err and counter < 1000:
        counter += 1
        phi = m.atan(2 * Aij_max / (A[i][i] - A[j][j])) / 2
        H = GetMatrixH(i, j, phi, len(A))
        A = np.dot(np.dot(H.T, A), H)
        if eig_is_changed:
            eigVectMatrix = np.dot(eigVectMatrix, H)
        else:
            eigVectMatrix = H
            eig_is_changed = True
        Aij_max, i, j = GetMaxValue(A)
    eigenvalues, eigenvectors = [], []
    for i in range(len(A)):
        eigenvalues.append(A[i][i])
        vector = []
        for j in range(len(A)):
            vector.append(eigVectMatrix[j][i])
        eigenvectors.append(vector)
    return eigenvalues, eigenvectors


# Функция, упорядочивающая собственные значения и вектора в порядке убывания значений
def GetOrderedFromYacobi(EVAL, EVECT):
    values, vectors = [], []
    for i in range(len(EVAL)):
        max, index = 0, 0
        for j in range(len(EVAL)):
            if EVAL[j] not in values and EVAL[j] > max:
                max = EVAL[j]
                index = j
        values.append(max)
        vectors.append(np.array(EVECT[index]))
    return values, vectors


def CreateStandartizedMatrix(data):
    # Стандартизация данных (нахождение мат. ожидания по каждой из компонент)
    standartized_vector = {val: data[val].mean() for val in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']}
    # Формирование стандартизированной матрицы
    Z = [data[val] - standartized_vector[val] for val in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']]
    return np.array(Z)


# Метод главных компонент
def PCA(data):
    # Создание стандартизированной матрицы (нахождение мат.ожидания по каждому из измерений, после чего - вычитание
    # от каждого измерения соответствующего мат. ожидания)
    Z = CreateStandartizedMatrix(data)
    # Создание матрицы ковариаций - матрицы, где на диагонали распологаются дисперсии соответствующих измерений, а на i,j пози-
    # циях - ковариации i,j - элементов (следовательно, матрица симметричная, так как cov(i,j) = cov(j,i))
    R = np.cov(Z)
    # Получение собственных векторов и собственных значений методом вращений Якоби
    eivenvalues_matrix, eigenvectors_matrix = YacobiMethod(R, 0.00001)
    # Упорядочивание значений по убыванию
    eigenvalues, eigenvectors = GetOrderedFromYacobi(eivenvalues_matrix, eigenvectors_matrix)

    # Согласно частному случаю отношения Рэлея для ковариационных матриц, направление максимальной дисперсии у проекции
    # всегда совпадает с айгенвектором, имеющим максимальное собственное значение, равное величине этой дисперсии.
    # Следовательно, нам нужно спроецировать нашу стандартизированную матрицу на собственные векторы с максимальными
    # собственными значениями, чтобы получить максимально наглядное и правильное представление результатов.

    # Отбрасывание всех собственных значений и соответствующих им собственных векторов, относительное количество информации
    # в которых меньше 1%
    s = sum(eigenvalues)
    index = 0
    while index < len(eigenvalues):
        if eigenvalues[index] / s < 0.01:
            eigenvalues.pop(index)
            eigenvectors.pop(index)
        else:
            index += 1
    v = -np.array(eigenvectors)
    # Перемножение матрицы полученных собственных векторов и стандартизированной матрицы для получения главных компонент
    P = np.dot(v, Z)
    return P

data_for_analys = pd.read_excel('forPCAanalysis.xlsx', header=1)
results = PCA(data_for_analys)
DrawGraphics(data_for_analys, results[:2])