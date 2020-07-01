
import math as m
import numpy as np



def seidel(A, b, eps):
    n = len(A)
    x = [.0 for i in range(n)]

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = m.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new

    return x

A=np.array([[4.503, 0.219, 0.527, 0.396],[0.259, 5.121, 0.423, 0.206],[0.413, 0.531, 4.317, 0.264],[0.327, 0.412, 0.203, 4.851]])
B=np.array([0.553, 0.358, 0.565, 0.436])
print('Исходная матрица\n')
print(A)
print('\nРезультаты вычислений:')
print('Метод Гаусса-Жордана: {}: '.format(seidel(A,B,0.001)))



