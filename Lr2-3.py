import numpy as np


def Inverse(A):
    return  np.linalg.inv(A)


Gilbert = list()
for i in range(1,4):
    Gilbert.append(list())
    for j in range(1,4):
        Gilbert[i-1].append(int(1/(i+j-1)*10000)/10000)

A = np.array(Gilbert)
print("Обратная матрица гильберта:\n {}".format(Inverse(A)))
print("Обратная матрица гильберта от обратной:\n {}".format(Inverse(Inverse(A))))
Gilbert_max = max(np.sum(A, axis =1))
Inverse_max = max(np.sum(Inverse(Inverse(A)),axis = 1))
print("Норма:\n {}".format(abs(Gilbert_max-Inverse_max)))
print("Число обусловленности:\n {}".format(np.linalg.cond(A)))