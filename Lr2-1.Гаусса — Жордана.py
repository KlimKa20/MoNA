import numpy
from typing import List
from typing import NoReturn


def zh_method(A_matr: List[list], free: list) -> numpy.ndarray:
    A_ext: numpy.ndarray = numpy.array(A_matr, dtype = float)
    A_ext = numpy.insert(A_ext, len(A_ext[0]), free, axis = 1)
    A_ext = settozero_ltriang(A_ext)
    A_ext = settozero_utriang(A_ext)
    return A_ext[:, len(A_ext[0]) - 1]


# при условии, что нет нулевых коэффициентов
def settozero_ltriang(A: numpy.ndarray) -> numpy.ndarray:
    matr: numpy.ndarray = numpy.array(A, dtype = float)
    for i in range(len(matr[0]) - 2):
        main_row: numpy.ndarray = numpy.array(matr[i], dtype = float)
        for j in range(i + 1, len(matr[0]) - 1):
            first_in_row: float = matr[j][i]
            for k in range(i, len(matr[0])):
                if main_row[k] != 0:
                    matr[j][k] = matr[j][k] + main_row[k] * (first_in_row / main_row[i]) * (-1)
        for k in range(i, len(matr[0])):
            if main_row[i] != 0:
                matr[i][k] = matr[i][k] / main_row[i]
    return matr


# при условии, что нет нулевых коэффициентов
def settozero_utriang(A: numpy.ndarray) -> numpy.ndarray:
    matr: numpy.ndarray = numpy.array(A, dtype = float)
    for i in range(len(matr[0]) - 2, 0, -1):
        main_row: numpy.ndarray = numpy.array(matr[i], dtype = float)
        for j in range(i - 1, - 1, -1):
            first_in_row: float = matr[j][i]
            for k in range(i, -1, -1):
                if main_row[k] != 0:
                    matr[j][k] = matr[j][k] + main_row[k] * (first_in_row / main_row[i]) * (-1)
            matr[j][len(matr[0]) - 1] = matr[j][len(matr[0]) - 1] + main_row[len(matr[0]) - 1] * \
                                        (first_in_row / main_row[i]) * (-1)
        for k in range(len(matr[0]) - 1, -1, -1):
            if main_row[i] != 0:
                matr[i][k] = matr[i][k] / main_row[i]
    return matr


def zh_main() -> NoReturn:
    # A: List[list] = [[4.503, 0.219, 0.527, 0.396], [0.259, 5.121, 0.423, 0.206], [0.413, 0.531, 4.317, 0.264],
    #                  [0.327, 0.412, 0.203, 4.851]]
    # b: list = [0.553, 0.358, 0.565, 0.436]
    A: List[list] = [[3.389, 0.273, 0.126, 0.418], [0.329, 2.796, 0.179, 0.278],
                     [0.186, 0.275, 2.987, 0.316], [0.197, 0.219, 0.274, 3.127]]
    b: list = [0.144, 0.297, 0.529, 0.869]
    # A: List[list] = [[115, -20, -75], [15, -50, -5], [6, 2, 20]]
    # b: list = [20, -40, 28]
    # A: List[list] = [[1, 1, 1], [4, 2, 1], [9, 3, 1]]
    # b: list = [0, 1, 3]
    print("Итог:", end = "\t")
    answer = zh_method(A, b)
    for _ in range(len(answer)):
        print("%s_%d: %f" % ('x', _, answer[_]), end = "\t")


if __name__ == "__main__":
    zh_main()
