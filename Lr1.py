import matplotlib.pyplot as plt
from matplotlib import rcParams
import math as m
import sympy as sym
import pandas as pd


#TASK 1
# a=0
# b=1
# f=lambda x:x-2*math.exp(-x)
# for i in range(0,3):
#     x1=(a+b)/2
#     if(f(a)*f(x1)<0):
#         b=x1
#     else:
#         a=x1
#     print("y={0} x={1}".format(f(x1),x1))

# x1=0
# x2=1
# f=lambda x:x-2*math.exp(-x)
# for i in range(0,3):
#     x0=x1-(x2-x1)/(f(x2)-f(x1))*f(x1)
#     if(f(x1)*f(x0)<0):
#         x2=x0
#     else:
#         x1=x0
#     print("y={0} x={1}".format(f(x0),x0))




# x1=1
# f=lambda x:x-2*math.exp(-x)
# f1=lambda x:1+2*math.exp(-x)
# for i in range(0,3):
#     x1=x1-f(x1)/f1(x1)
#     print("y={0} x={1}".format(f(x1),x1))


#TASK2
# def Solve(Tc, Th, Jc, Jh, i):
#     if i >= 100:
#         return Tc, Th, Jc, Jh
#     next_Tc = ((Jc - 17.41 * Tc + 5188.18) / 5.67e-8) ** (1 / 4)
#     next_Th = ((2250 + Jh - 1.865 * Th) / 5.67e-8) ** (1 / 4)
#     next_Jc = 2352.71 + 0.71 * Jh - 7.46 * Tc
#     next_Jh = 11093 + 0.71 * Jc - 7.46 * Th
#     return Solve(next_Tc, next_Th, next_Jc, next_Jh, i + 1)
#
#
# print(Solve(298, 298, 3000, 5000, 0))
#
# x = [i for i in range(0, 100)]
# Tc = []
# Th = []
# Jc = []
# Jh = []
#
# for i in x:
#     Tc.append(abs(Solve(298, 298, 3000, 5000, i)[0] - 481))
#     Th.append(abs(Solve(298, 298, 3000, 5000, i)[1] - 671))
#     Jc.append(abs(Solve(298, 298, 3000, 5000, i)[2] - 6222))
#     Jh.append(abs(Solve(298, 298, 3000, 5000, i)[3] - 10504))
# x.reverse()
# plt.plot(x, Tc,color="red")
# plt.show()
# plt.plot(x, Th,color="green")
# plt.show()
# plt.plot(x, Jc,color="brown")
# plt.show()
# plt.plot(x, Jh,color="black")
# plt.show()




#Task4
# def MyNewton( x):
#     MaxError = 0.000001
#     for i in range(100):
#         next_x = x - MyFunction(x) / MyDerivative(x)
#         if m.fabs((next_x - x) / x) < MaxError:
#             return next_x, i
#         else:
#             x = next_x
#     else:
#         raise RuntimeError('Выбрано плохое начальное приближение, достигнут лимит итераций!')
#
# def MyFunction(x):
#     return 2*x**3 - 4*x**2 -4*x - 20
# def MyDerivative(x):
#     return 6*x**2-8*x-4
#
#
#
# casat = lambda x: MyFunction(4) + MyDerivative(4) * (x - 4)
# x, y, y_cas = [], [], []
# for i in range(1000):
#     x.append(3 + 1 / (i + 1))
#     y.append(MyFunction(3 + 1 / (i + 1)))
#     y_cas.append(casat(3 + 1 / (i + 1)))
#
# plt.plot(x, [0 for i in range(1000)])
# plt.plot(x, y, label=u'График функции f(x)')
# plt.plot(x, y_cas, label=u'Касательная к f(x) в точке 4')
# plt.grid(True)
# plt.xlabel(u'X')
# plt.ylabel(u'Y')
# plt.title(u'Графическая иллюстрация')
# plt.legend()
#
# print(MyNewton(4))





#TASK 5
# def MyIteratio(x,eps):
#     result_X, result_Y, delta_X, delta_Y = [x], [MyFunction(x)], [0], [0]
#     diff= MyDerivative(x)
#     for i in range(100):
#         next_x = x - float(MyFunction(x)) /diff
#         result_X.append(next_x)
#         result_Y.append((MyFunction(next_x)))
#         delta_X.append(next_x - x)
#         delta_Y.append((MyFunction(next_x) - MyFunction(x)))
#         if m.fabs((next_x - x) / x) < eps:
#             print('\nMyIteration results:\n')
#             PrintTable(result_X, result_Y, delta_X, delta_Y)
#             return next_x
#         else:
#             x = next_x
#     else:
#         raise ("Bad input")
#
# def PrintTable(X, Y, delta_x, delta_y):
#     df = pd.DataFrame({
#                 'X': X,
#                 'Y': Y,
#                 'DeltaX': delta_x,
#                 'DeltaY': delta_y
#                 })
#     print(df)
# def MyNewton(x,eps):
#     result_X,result_Y,delta_X,delta_Y = [x],[MyFunction(x)],[0],[0]
#
#     for i in range(100):
#         next_x = x - float(MyFunction(x)) / MyDerivative(x)
#         result_X.append(next_x)
#         result_Y.append((MyFunction(next_x)))
#         delta_X.append(next_x-x)
#         delta_Y.append((MyFunction(next_x)-MyFunction(x)))
#         if m.fabs((next_x - x) / x) < eps:
#             print('\nMyNewton results:\n')
#             PrintTable(result_X, result_Y, delta_X, delta_Y)
#             return next_x
#         else:
#             x=next_x
#     else:
#         raise ("Bad input")
# def MyFunction(x):
#     return 4*(1+m.sqrt(x))*m.log10(x)-1
# def MyDerivative(x):
#     return (4/m.sqrt(x)+4/x)+2*(m.log10(x)/m.sqrt(x))
# MyIteratio(1,.00001)
# MyNewton(1,0.00001)




# #TASK6
# def Jacobian(v_str, f_list):
#     vars = sym.symbols(v_str)
#     f = sym.sympify(f_list)
#     J = sym.zeros(len(f), len(vars))
#     for i, fi in enumerate(f):
#         for j, s in enumerate(vars):
#             J[i, j] = sym.diff(fi, s)
#     return J
#
#
# def SistRoots(functions, vars: dict):
#     jacobi_matrix = Jacobian(' '.join(list(vars.keys())), functions)
#     jacobian = lambda vars_dict: float(jacobi_matrix.subs(list(vars_dict.items())).det())
#     print(jacobian(vars))
#
#
# def SystNewton(functions, vars: dict, ErrMax):
#     func_matrix = sym.Matrix(functions)
#     jacobi_matrix = Jacobian(' '.join(list(vars.keys())), functions)
#     Xi = sym.Matrix(list(vars.values()))
#     count_jacobi = lambda vars: jacobi_matrix.subs(list(vars.items()))
#     x_list, y_list, delt_x, delt_y = [vars['x']], [vars['y']], [0], [0]
#     for i in range(100):
#         delta_x = -count_jacobi(vars).inv() * func_matrix.subs(list(vars.items()))
#         delt_x.append(float(delta_x[0]))
#         delt_y.append(float(delta_x[1]))
#         Xi = delta_x + Xi
#         vars = {'x': float(Xi[0]), 'y': float(Xi[1])}
#         x_list.append(vars['x'])
#         y_list.append(vars['y'])
#         max_delta = max(m.fabs(float(delta_x[0])), m.fabs(float(delta_x[1])))
#         if max_delta < ErrMax:
#             df = pd.DataFrame({
#                 'X': x_list,
#                 'Y': y_list,
#                 'DeltaX': delt_x,
#                 'DeltaY': delt_y
#             })
#             print(df)
#             return vars
#     else:
#         raise RuntimeError('Выбрано плохое начальное приближение, достигнут лимит итераций!')
#
# functions = ['sin(x+y) + 1.2*x - 1', 'x**2 + y**2 - 1']
# vars = {'x': 40, 'y': 26}
# print('\nAnswer: {}'.format(SystNewton(functions, vars, 0.000001)))