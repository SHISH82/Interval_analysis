from itertools import combinations, product

import numpy as np


TOL = 1e-12
RANK_TOL = 1e-12


def get_matrices():
    a1 = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00],
    ])
    a2 = np.array([
        [1.10, 0.90, 1.10],
        [1.40, 1.00, 0.80],
        [0.80, 1.40, 1.20],
    ])
    tomo = np.ones((3, 2))
    regress = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    square = np.ones((3, 3))

    return [
        ("A1 — томография", a1, tomo),
        ("A1 — регрессия", a1.copy(), regress),
        ("A2 — квадратная матрица", a2, square),
    ]


def interval_bounds(center, radius, delta):
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("Параметр delta должен быть конечным и неотрицательным.")
    if center.shape != radius.shape:
        raise ValueError("Размеры центра и матрицы радиусов должны совпадать.")
    if np.any(radius < 0):
        raise ValueError("Радиусы не могут быть отрицательными.")

    lower = center - delta * radius
    upper = center + delta * radius
    return lower, upper


def contains(matrix, lower, upper, tol=TOL):
    if matrix.shape != lower.shape or matrix.shape != upper.shape:
        return False
    return bool(np.all((matrix >= lower - tol) & (matrix <= upper + tol)))


def matrix_rank(matrix, tol=RANK_TOL):
    values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(values > tol))


def coefficient_interval(center, radius, delta, tol=TOL):
    lower, upper = interval_bounds(center, radius, delta)
    if np.any(lower <= 0):
        raise ValueError("Для этого расчёта нужны положительные интервалы.")

    left = np.max(lower[:, 0] / upper[:, 1])
    right = np.min(upper[:, 0] / lower[:, 1])
    if left > right + tol:
        return None
    if left > right:
        left = right = (left + right) / 2
    return float(left), float(right)


def solve_rectangular(center, radius):
    if center.shape != (3, 2) or radius.shape != (3, 2):
        raise ValueError("Ожидаются матрицы размера 3 x 2.")
    if not np.all(center[:, 1] == 1) or not np.all(radius[:, 0] == 1):
        raise ValueError("Нужны единицы во втором столбце центра и первом столбце радиусов.")

    a_min = np.min(center[:, 0])
    a_max = np.max(center[:, 0])
    if np.all(radius[:, 1] == 1):
        delta = (a_max - a_min) / (a_max + a_min + 2)
    elif np.all(radius[:, 1] == 0):
        delta = (a_max - a_min) / 2
    else:
        raise ValueError("Второй столбец радиусов должен состоять из нулей или единиц.")

    interval = coefficient_interval(center, radius, delta)
    if interval is None:
        raise ValueError("Общий коэффициент не найден.")
    k = (interval[0] + interval[1]) / 2

    lower, upper = interval_bounds(center, radius, delta)
    left = np.maximum(lower[:, 1], lower[:, 0] / k)
    right = np.minimum(upper[:, 1], upper[:, 0] / k)
    if np.any(left > right + TOL):
        raise ValueError("Не удалось построить матрицу.")

    second = np.clip((left + right) / 2, lower[:, 1], upper[:, 1])
    matrix = np.column_stack((k * second, second))
    return float(delta), float(k), matrix


def rectangular_minors(matrix):
    return np.array([
        matrix[i, 0] * matrix[j, 1] - matrix[j, 0] * matrix[i, 1]
        for i, j in combinations(range(len(matrix)), 2)
    ])


def solve_square(center, radius):
    if center.shape != (3, 3) or radius.shape != (3, 3):
        raise ValueError("Ожидаются матрицы размера 3 x 3.")
    if not np.all(radius == 1):
        raise ValueError("Матрица радиусов должна состоять из единиц.")
    if matrix_rank(center) < 3:
        raise ValueError("Матрица центров должна быть невырожденной.")

    best_norm = 0.0
    best_x = None
    best_y = None

    for signs in product((-1.0, 1.0), repeat=3):
        y = np.array(signs)
        x = np.linalg.solve(center, y)
        norm = np.linalg.norm(x, ord=1)
        if norm > best_norm:
            best_norm = norm
            best_x = x
            best_y = y

    delta = 1 / best_norm
    direction = np.sign(best_x)
    direction[np.abs(best_x) <= TOL] = 0
    matrix = center - delta * np.outer(best_y, direction)
    return float(delta), best_x, best_y, matrix


def print_square(name, center, radius):
    delta, x, y, matrix = solve_square(center, radius)
    lower, upper = interval_bounds(center, radius, delta)
    belongs = contains(matrix, lower, upper)
    rank = matrix_rank(matrix)
    determinant = np.linalg.det(matrix)
    residual = np.linalg.norm(matrix @ x, ord=np.inf) / np.linalg.norm(x, ord=1)

    print(name)
    print("Матрица центров:")
    print(center)
    print("Множители радиусов:")
    print(radius)
    print(f"Определитель центра: {np.linalg.det(center):.12f}")
    print(f"Максимальная сумма модулей x: {np.linalg.norm(x, ord=1):.12f}")
    print(f"delta_min ≈ {delta:.12f}")
    print("Диапазон: [delta_min, +∞)")
    print("Матрица радиусов при delta_min:")
    print(delta * radius)
    print("Вектор y:")
    print(y)
    print("Вектор x:")
    print(x)
    print("Точечная матрица:")
    print(matrix)
    print("Принадлежит интервалам:", "да" if belongs else "нет")
    print(f"Ранг: {rank}")
    print(f"Определитель: {determinant:.3e}")
    print(f"Нормированная невязка A x: {residual:.3e}")
    valid = belongs and rank == 2 and abs(determinant) <= TOL and residual <= TOL
    print("Проверки пройдены:", "да" if valid else "нет")
    print()


def main():
    np.set_printoptions(precision=12, suppress=True)
    print(f"Допуск проверок: {TOL:g}")
    print(f"Порог ранга: {RANK_TOL:g}")
    print()

    for name, center, radius in get_matrices()[:2]:
        delta, k, matrix = solve_rectangular(center, radius)
        lower, upper = interval_bounds(center, radius, delta)
        belongs = contains(matrix, lower, upper)
        rank = matrix_rank(matrix)
        residual = np.max(np.abs(matrix[:, 0] - k * matrix[:, 1]))
        minors = rectangular_minors(matrix)

        print(name)
        print("Матрица центров:")
        print(center)
        print("Множители радиусов:")
        print(radius)
        print(f"delta_min ≈ {delta:.12f}")
        print("Диапазон: [delta_min, +∞)")
        print("Матрица радиусов при delta_min:")
        print(delta * radius)
        print(f"k = {k:.12f}")
        print("Точечная матрица:")
        print(matrix)
        print("Принадлежит интервалам:", "да" if belongs else "нет")
        print(f"Ранг: {rank}")
        print(f"Невязка пропорциональности: {residual:.3e}")
        print("Миноры по строкам (1, 2), (1, 3), (2, 3):")
        print(" ".join(f"{value:.3e}" for value in minors))
        valid = belongs and rank == 1 and residual <= TOL and np.all(np.abs(minors) <= TOL)
        print("Проверки пройдены:", "да" if valid else "нет")
        print()

    print_square(*get_matrices()[2])


if __name__ == "__main__":
    main()
