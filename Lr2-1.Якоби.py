import numpy
from typing import NoReturn
from typing import List


def jacobi_method(A_matr: List[list], free: list) -> numpy.ndarray:
    # оператор
    A: numpy.ndarray = numpy.array(A_matr, dtype = float)
    check_diag_dominance(A)
    # свободные члены
    b: numpy.ndarray = numpy.array(free, dtype = float)
    # D^(-1) = E * (1 / a_ii), i = 1 -> n
    D_rev: numpy.ndarray = create_D_rev(A)
    # d = D^(-1) * b
    d = numpy.dot(D_rev, b)
    # нижний треугольник А
    L: numpy.ndarray = numpy.tril(A, -1)
    # верхний треугольник А
    U: numpy.ndarray = numpy.triu(A, 1)
    # B = D^(-1) * (L + U)
    B: numpy.ndarray = numpy.dot(-D_rev, (L + U))

    for _ in range(len(B[0])):
        print("{:>9s}{:d}".format('x_', _), end = "\t")
    print()
    answer: numpy.ndarray = iterations(B, d)
    return answer


def check_diag_dominance(matr: numpy.ndarray) -> NoReturn:
    for i in range(len(matr)):
        tmp = [abs(el) for el in matr[i]]
        if abs(matr[i][i]) <= sum(tmp) - abs(matr[i][i]):
            raise ValueError


def create_D_rev(A_matr: numpy.ndarray) -> numpy.ndarray:
    D: numpy.ndarray = numpy.diag(A_matr)
    D: numpy.ndarray = numpy.diag(D)
    for i in range(len(A_matr)):
        D[i][i] = 1 / D[i][i]
    return D


def iterations(B: numpy.ndarray, d: numpy.ndarray) -> numpy.ndarray:
    cur: numpy.ndarray = numpy.array(d, dtype = float)
    prev: numpy.ndarray = numpy.array([])
    while absolute_err_exit(cur, prev, 0.001):
        prev = numpy.array(cur, dtype = float)
        cur = numpy.array([0.0 for _ in range(len(d))])
        for i in range(len(B)):
            for j in range(len(B)):
                cur[i] += B[i][j] * prev[j]
            cur[i] += d[i]
        for _ in cur:
            print("{:10.5f}".format(_), end = "\t")
        print()
    return cur


def absolute_err_exit(cur: numpy.ndarray, prev: numpy.ndarray, prec = 0.001) -> bool:
    if len(prev) == 0:
        return True
    for item1, item2 in zip(cur, prev):
        if abs(item1 - item2) <= prec:
            return False
    return True


def j_main() -> NoReturn:
    # A: List[list] = [[4.503, 0.219, 0.527, 0.396], [0.259, 5.121, 0.423, 0.206], [0.413, 0.531, 4.317, 0.264],
    #                  [0.327, 0.412, 0.203, 4.851]]
    # b: list = [0.553, 0.358, 0.565, 0.436]
    A: List[list] = [[3.389, 0.273, 0.126, 0.418], [0.329, 2.796, 0.179, 0.278],
                     [0.186, 0.275, 2.987, 0.316], [0.197, 0.219, 0.274, 3.127]]
    b: list = [0.144, 0.297, 0.529, 0.869]
    # A: List[list] = [[115, -20, -75], [15, -50, -5], [6, 2, 20]]
    # b: list = [20, -40, 28]
    answer = jacobi_method(A, b)
    print("Итог:", end = "\t")
    for _ in range(len(answer)):
        print("%s_%d: %f" % ('x', _, answer[_]), end = "\t")


if __name__ == "__main__":
    j_main()
