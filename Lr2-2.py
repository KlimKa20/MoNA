import numpy as np


def gr_descend(A, b, n):
    x = np.array([0.5] * n)
    r = b - np.transpose(np.dot(A, np.transpose(x)))
    z = r
    n = 0
    while np.linalg.norm(r) / np.linalg.norm(b) > 0.001:
        alpha = np.dot(r, np.transpose(r)) / np.dot(np.dot(A, np.transpose(z)), z)
        x = x + alpha * z
        r1 = r
        r = r - alpha * np.transpose(np.dot(A, np.transpose(z)))
        beta = np.dot(r, np.transpose(r)) / np.dot(r1, np.transpose(r1))
        z = r + beta * z
        n += 1
    print("Сошёлся {} итераций".format(n))
    return x


def jacobi(a_matrix: np.matrix, residual: np.matrix) -> np.matrix:
    _to_return = {}
    if 'inverted' not in _to_return:
        _to_return['inverted'] = np.linalg.inv(np.diag(np.diag(a_matrix)))
    return np.matrix(_to_return['inverted'] * residual)


def solve(A: np.matrix, b_vec: np.vstack, n, max_iter: int = 200):
    i = 0
    x = np.empty(n)
    x_vec = np.vstack(x)
    tolerance = 1e-5
    residual_values = []
    finished_iter = 0
    residual = b_vec - A * x_vec
    div = jacobi(A, residual)
    delta_new = residual.T * div

    while i < max_iter and np.linalg.norm(residual) > tolerance:
        q_vec = A * div
        alpha = delta_new / (div.T * q_vec)
        x_vec = x_vec + alpha * div
        residual = residual - alpha * q_vec
        s_pre = jacobi(A, residual)
        delta_old = delta_new
        delta_new = residual.T * s_pre
        beta = delta_new / delta_old
        div = s_pre + beta * div
        residual_values.append(np.linalg.norm(residual))
        i += 1
    finished_iter = i
    print("Сошёлся {} итераций".format(i))
    return x_vec, residual


a = np.array([[3.389, 0.273, 0.126, 0.418],
              [0.329, 2.796, 0.179, 0.287],
              [0.186, 0.275, 2.987, 0.316],
              [0.197, 0.219, 0.274, 3.127]])
b = np.array([0.144, 0.297, 0.529, 0.869])
gr_descend(a, b, 4)
