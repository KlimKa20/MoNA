import numpy as np
import matplotlib.pyplot as plt
from math import log, sin, cos, exp
import math


def right_derivative(f, x, n, delta):
    if n == 1:
        return (f(x + delta) - f(x)) / delta
    return (right_derivative(f, x + delta, n - 1, delta) - right_derivative(f, x, n - 1, delta)) / delta


def left_derivative(f, x, n, delta):
    if n == 1:
        return (f(x) - f(x - delta)) / delta
    return (left_derivative(f, x, n - 1, delta) - left_derivative(f, x - delta, n - 1, delta)) / delta


def derivative(f, x, n, delta):
    if n == 1:
        return (f(x + delta) - f(x - delta)) / (2 * delta)
    return (derivative(f, x + delta, n - 1, delta) - derivative(f, x - delta, n - 1, delta)) / (2 * delta)


z1 = [0.25, 0.51, 0.99, 1.09, 1.89, 2.39]
delta = 0.001


#
def f1(x):
    return exp(2 * x) * cos(3 * x)


print('e^(2x)*cos(3x)')
print("f(x)': 0.25) -0.9587949645")
print("f(x)': 0.51) -8.0864523832")
print("f(x)'': 0.99) 20.8413434406")
print("f(x)'': 1.09) 57.4611173972")
print("f(x)''': 1.89) -1421.4099360944")
print("f(x)''': 2.39) 4292.7866586904")
for dot in z1[:2]:
    print('Производные первого порядка в точках 0.25 и 0.51:')
    print('Левая производная в точке ', dot, ':', left_derivative(f1, dot, 1, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f1, dot, 1, delta))
    print('Производная в точке ', dot, ':', derivative(f1, dot, 1, delta))
    print("//////////////////////////////////////////////////////////////////")
for dot in z1[2:4]:
    print('Производные второго порядка в точках 0.99 и 1.09:')
    print('Левая производная в точке ', dot, ':', left_derivative(f1, dot, 2, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f1, dot, 2, delta))
    print('Производная в точке ', dot, ':', derivative(f1, dot, 2, delta))
    print("//////////////////////////////////////////////////////////////////")
for dot in z1[4:6]:
    print('Производные второго порядка в точках 1.89 и 2.39:')
    print('Левая производная в точке ', dot, ':', left_derivative(f1, dot, 3, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f1, dot, 3, delta))
    print('Производная в точке ', dot, ':', derivative(f1, dot, 3, delta))
    print("//////////////////////////////////////////////////////////////////")
print()

z2 = [0.25, 0.51, 0.99, 1.09, 1.89, 2.39]


def f2(x):
    return sin(log(x))


print('sin(ln(x))')
print("f(x)': 0.25) 0.733827899")
print("f(x)': 0.51) 1.5328242453")
print("f(x)'': 0.99) -1.0099982952")
print("f(x)'': 1.09) -0.9110008147")
print("f(x)''': 1.89) 0.383257439")
print("f(x)''': 2.39) 0.2153048472")
for dot in z2[:2]:
    print('Производные первого порядка в точках 0.25 и 0.51:')
    print('Левая производная в точке ', dot, ':', left_derivative(f2, dot, 1, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f2, dot, 1, delta))
    print('Производная в точке ', dot, ':', derivative(f2, dot, 1, delta))
    print("//////////////////////////////////////////////////////////////////")
for dot in z2[2:4]:
    print('Производные второго порядка в точках 0.99 и 1.09:')
    print('Левая производная в точке ', dot, ':', left_derivative(f2, dot, 2, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f2, dot, 2, delta))
    print('Производная в точке ', dot, ':', derivative(f2, dot, 2, delta))
    print("//////////////////////////////////////////////////////////////////")
for dot in z2[4:6]:
    print('Производные второго порядка в точках 1.89 и 2.39:')
    print('Левая производная в точке ', dot, ':', left_derivative(f2, dot, 3, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f2, dot, 3, delta))
    print('Производная в точке ', dot, ':', derivative(f2, dot, 3, delta))
    print("//////////////////////////////////////////////////////////////////")
print()

z3 = [0.25, 0.51, 0.99, 1.09, 1.89, 2.39]

def f3(x):
    return log(x)


print('ln(x)')
print("f(x)': 0.25) 4")
print("f(x)': 0.51) 1.96078431372")
print("f(x)'': 0.99) -1.0203040506")
print("f(x)'': 1.09) -0.84167999326")
print("f(x)''': 1.89) 0.14812030152")
print("f(x)''': 2.39) 0.07324977535")
for dot in z3[:2]:
    print('Производные первого порядка в точках 0.25 и 0.51:')
    print('Левая производная в точке ', dot, ':', left_derivative(f3, dot, 1, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f3, dot, 1, delta))
    print('Производная в точке ', dot, ':', derivative(f3, dot, 1, delta))
    print("//////////////////////////////////////////////////////////////////")
for dot in z3[2:4]:
    print('Производные второго порядка в точках 0.99 и 1.09:')
    print('Левая производная в точке ', dot, ':', left_derivative(f3, dot, 2, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f3, dot, 2, delta))
    print('Производная в точке ', dot, ':', derivative(f3, dot, 2, delta))
    print("//////////////////////////////////////////////////////////////////")
for dot in z3[4:6]:
    print('Производные второго порядка в точках 1.89 и 2.39:')
    print('Левая производная в точке ', dot, ':', left_derivative(f3, dot, 3, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f3, dot, 3, delta))
    print('Производная в точке ', dot, ':', derivative(f3, dot, 3, delta))
    print("//////////////////////////////////////////////////////////////////")
print()

z4 = [0.25, 0.51, 0.99, 1.09, 1.89, 2.39]


def f4(x):
    return cos(x) + sin(x)


print('cos(x) + sin(x)')
print("f(x)': 0.25) 0.7215084625")
print("f(x)': 0.51) 0.3845672608")
print("f(x)'': 0.99) -1.3847158392")
print("f(x)'': 1.09) -1.3491122813")
print("f(x)''': 1.89) 1.2632961742")
print("f(x)''': 2.39) 1.4134055489")
for dot in z4[:2]:
    print('Производные первого порядка в точках 0.25 и 0.51:')
    print('Левая производная в точке ', dot, ':', left_derivative(f4, dot, 1, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f4, dot, 1, delta))
    print('Производная в точке ', dot, ':', derivative(f4, dot, 1, delta))
    print("//////////////////////////////////////////////////////////////////")
for dot in z4[2:4]:
    print('Производные второго порядка в точках 0.99 и 1.09:')
    print('Левая производная в точке ', dot, ':', left_derivative(f4, dot, 2, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f4, dot, 2, delta))
    print('Производная в точке ', dot, ':', derivative(f4, dot, 2, delta))
    print("//////////////////////////////////////////////////////////////////")
for dot in z4[4:6]:
    print('Производные второго порядка в точках 1.89 и 2.39:')
    print('Левая производная в точке ', dot, ':', left_derivative(f4, dot, 3, delta))
    print('Правая производная в точке ', dot, ':', right_derivative(f4, dot, 3, delta))
    print('Производная в точке ', dot, ':', derivative(f4, dot, 3, delta))
    print("//////////////////////////////////////////////////////////////////")
print()
