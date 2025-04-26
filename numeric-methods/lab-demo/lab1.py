import numpy as np
import plotly.graph_objs as go
import streamlit as st
from scipy.linalg import hessenberg, qr

import streamlit_elements as st_elements
import utils


def solve_lu(a):
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath}
\begin{document}
\section*{Решение системы}
    """
    A = a[:, :-1]
    b = a[:, -1]
    n = A.shape[0]

    st.subheader("Исходная матрица и вектор b:")
    st.latex(r"A = " + utils.matrix_to_latex(A))
    st.latex(r"b = " + utils.matrix_to_latex(b))
    st.divider()

    latex_content += r"\subsection*{Исходная матрица и вектор b:}" + "\n"
    latex_content += r"\[ A = " + utils.matrix_to_latex(A) + r" \]" + "\n"
    latex_content += r"\[ b = " + utils.matrix_to_latex(b) + r" \]" + "\n"

    # LU-разложение
    st.subheader("LU-разложение")
    try:
        U = A.copy().astype(float)
        L = np.eye(n)
        P = np.eye(n)
        pivot_amount = 0

        for k in range(n - 1):
            # Частичный выбор ведущего элемента
            pivot = np.argmax(np.abs(U[k:, k])) + k
            if pivot != k:
                # Перестановка строк в U
                U[[k, pivot], k:] = U[[pivot, k], k:]

                # Перестановка строк в L
                if k > 0:
                    L[[k, pivot], :k] = L[[pivot, k], :k]

                # Перестановка строк в P
                P[[k, pivot], :] = P[[pivot, k], :]
                pivot_amount += 1

            if U[k, k] == 0:
                raise ValueError("Матрица вырождена")

            # Исключение Гаусса
            for i in range(k + 1, n):
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] -= L[i, k] * U[k, k:]

        b_permuted = P @ b
    except np.linalg.LinAlgError:
        st.error("Матрица вырождена, LU-разложение невозможно")
        return None

    st.latex(r"L = " + utils.matrix_to_latex(L))
    st.latex(r"U = " + utils.matrix_to_latex(U))
    st.divider()

    latex_content += r"\subsection*{LU-разложение}" + "\n"
    latex_content += r"\[ L = " + utils.matrix_to_latex(L) + r" \]" + "\n"
    latex_content += r"\[ U = " + utils.matrix_to_latex(U) + r" \]" + "\n"

    # Решение Lz = b (прямая подстановка)
    st.subheader("Решение системы Lz = b")
    latex_content += r"\subsection*{Решение системы Lz = b (прямая подстановка)}" + "\n"
    z = np.zeros(n)
    for i in range(n):
        sum_terms = []
        sum_value = 0.0
        for k in range(i):
            coeff = L[i, k]
            if abs(coeff) > 1e-10:  # Игнорируем нулевые коэффициенты
                sum_terms.append(f"{utils.format_number(coeff)} \cdot z_{k+1}")
                sum_value += coeff * z[k]

        z[i] = b_permuted[i] - sum_value
        formula = rf"z_{i+1} = b_{i+1}"
        if sum_terms:
            formula += rf" - ({' + '.join(sum_terms)})"

        st.latex(formula + f" = {utils.format_number(z[i])}")
        latex_content += rf"\[ {formula} = {utils.format_number(z[i])} \]" + "\n"

    st.latex(r"z = " + utils.matrix_to_latex(z))
    st.divider()

    latex_content += r"\[ z = " + utils.matrix_to_latex(z) + r" \]" + "\n"

    # Решение Ux = z (обратная подстановка)
    st.subheader("Решение системы Ux = z")
    latex_content += (
        r"\subsection*{Решение системы Ux = z (обратная подстановка)}" + "\n"
    )
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_terms = []
        sum_value = 0.0
        for k in range(i + 1, n):
            coeff = U[i, k]
            if abs(coeff) > 1e-10:
                sum_terms.append(f"{utils.format_number(coeff)} \cdot x_{k+1}")
                sum_value += coeff * x[k]

        x[i] = (z[i] - sum_value) / U[i, i]
        formula = rf"x_{i+1} = \frac{{z_{i+1}"
        if sum_terms:
            formula += rf" - ({' + '.join(sum_terms)})"
        formula += rf"}}{{{utils.format_number(U[i,i])}}}"

        st.latex(formula + f" = {utils.format_number(x[i])}")
        latex_content += rf"\[ {formula} = {utils.format_number(x[i])} \]" + "\n"

    st.latex(r"x = " + utils.matrix_to_latex(x))
    st.divider()

    latex_content += r"\[ x = " + utils.matrix_to_latex(x) + r" \]" + "\n"

    # Дополнительные вычисления (определитель, обратная матрица)
    sign = (-1) ** pivot_amount  # Учет перестановок строк
    det_lu = sign * np.prod(np.diag(U))
    st.subheader("Определитель матрицы")
    st.latex(r"\det(A) = " + utils.format_number(det_lu))
    st.divider()

    latex_content += r"\subsection*{Определитель матрицы}" + "\n"
    latex_content += rf"\[ \det(A) = {utils.format_number(det_lu)} \]" + "\n"

    # 1. Проверка невязки Ax - b
    st.subheader("Проверка невязки")
    latex_content += r"\subsection*{Проверка невязки}" + "\n"

    residual = (P @ A) @ x - (P @ b)
    residual_norm = np.linalg.norm(residual)

    st.latex(r"Ax - b = " + utils.matrix_to_latex(residual))
    st.latex(r"\|Ax - b\| = " + f"{residual_norm:.2e}")

    latex_content += r"\[ Ax - b = " + utils.matrix_to_latex(residual) + r" \]" + "\n"
    latex_content += rf"\[ \|Ax - b\| = {residual_norm:.2e} \]" + "\n"

    if residual_norm < 1e-8:
        st.success("Невязка минимальна, решение корректно!")
        latex_content += r"\textbf{Невязка минимальна, решение корректно!}" + "\n"
    else:
        st.warning("Обнаружена значительная невязка!")
        latex_content += r"\textbf{Обнаружена значительная невязка!}" + "\n"
    st.divider()

    # 2. Проверка обратной матрицы
    st.subheader("Проверка обратной матрицы")
    latex_content += r"\subsection*{Проверка обратной матрицы}" + "\n"

    try:
        A_inv = np.linalg.inv(A)
        I_calculated = A @ A_inv
        I_error = np.linalg.norm(I_calculated - np.eye(n))

        st.latex(r"A^{-1} = " + utils.matrix_to_latex(A_inv))
        st.latex(r"A \cdot A^{-1} = " + utils.matrix_to_latex(I_calculated))
        st.latex(r"\|A \cdot A^{-1} - I\| = " + f"{I_error:.2e}")

        latex_content += r"\[ A^{-1} = " + utils.matrix_to_latex(A_inv) + r" \]" + "\n"
        latex_content += (
            r"\[ A \cdot A^{-1} = "
            + utils.matrix_to_latex(I_calculated)
            + r" \]"
            + "\n"
        )
        latex_content += rf"\[ \|A \cdot A^{-1} - I\| = {I_error:.2e} \]" + "\n"

        if I_error < 1e-8:
            st.success("Обратная матрица вычислена верно!")
            latex_content += r"\textbf{Обратная матрица вычислена верно!}" + "\n"
        else:
            st.warning("Возможная ошибка в вычислении обратной матрицы!")
            latex_content += (
                r"\textbf{Возможная ошибка в вычислении обратной матрицы!}" + "\n"
            )

    except np.linalg.LinAlgError:
        st.error("Матрица вырождена, обратной матрицы не существует")
        latex_content += (
            r"\textbf{Матрица вырождена, обратной матрицы не существует}" + "\n"
        )
    st.divider()

    # 3. Проверка определителя
    st.subheader("Проверка определителя")
    latex_content += r"\subsection*{Проверка определителя}" + "\n"

    det_numpy = np.linalg.det(A)  # Эталонное значение
    rel_error = (
        abs(det_lu - det_numpy) / abs(det_numpy) if det_numpy != 0 else float("inf")
    )

    st.latex(r"\det(A)_{\text{LU}} = " + utils.format_number(det_lu))
    st.latex(r"\det(A)_{\text{numpy}} = " + utils.format_number(det_numpy))
    st.latex(r"Относительная\ ошибка = " + f"{rel_error:.2e}")

    latex_content += (
        rf"\[ \det(A)_{{\text{{LU}}}} = {utils.format_number(det_lu)} \]" + "\n"
    )
    latex_content += (
        rf"\[ \det(A)_{{\text{{numpy}}}} = {utils.format_number(det_numpy)} \]" + "\n"
    )
    latex_content += rf"\[ \text{{Относительная ошибка}} = {rel_error:.2e} \]" + "\n"

    if rel_error < 1e-8:
        st.success("Определитель вычислен верно!")
        latex_content += r"\textbf{Определитель вычислен верно!}" + "\n"
    else:
        st.warning("Возможная ошибка в вычислении определителя!")
        latex_content += r"\textbf{Возможная ошибка в вычислении определителя!}" + "\n"

    latex_content += r"\end{document}"
    pdf_bytes = utils.compile_latex_to_pdf(latex_content)
    return pdf_bytes


def solve_thomas(A):
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{a4paper, margin=1.5cm}

\begin{document}"""

    st.subheader("Исходная система:")
    st.latex(utils.matrix_to_latex(A, line=True))
    latex_content += r"""
\section*{Решение трёхдиагональной системы методом прогонки}
\subsection*{Исходная система:}
"""
    latex_content += rf"\[ {utils.matrix_to_latex(A, line=True)} \]"

    n = A.shape[0]

    # Извлекаем диагонали
    st.subheader("Шаг 1: Извлечение диагоналей")
    latex_content += "\subsection*{Шаг 1: Извлечение диагоналей}"

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        for j in range(n + 1):
            if j < n:
                if i == j:
                    b[i] = A[i, j]
                elif i == j + 1:
                    a[i] = A[i, j]
                elif i == j - 1:
                    c[i] = A[i, j]
                elif A[i, j] != 0:
                    raise ValueError("Матрица не трёхдиагональная!")
            else:
                d[i] = A[i, j]

    # Добавляем информацию о диагоналях
    st.latex(
        f"\\text{{Нижняя диагональ }} (a_i): {np.array2string(a[1:], separator=', ')}"
    )
    st.latex(
        f"\\text{{Главная диагональ }} (b_i): {np.array2string(b, separator=', ')}"
    )
    st.latex(
        f"\\text{{Верхняя диагональ }} (c_i): {np.array2string(c[:-1], separator=', ')}"
    )
    st.latex(f"\\text{{Правые части }} (d_i): {np.array2string(d, separator=', ')}")

    latex_content += r"\begin{align*}" + "\n"
    latex_content += (
        f"\\text{{Нижняя диагональ }} (a_i) &: {np.array2string(a[1:], separator=', ')} \\\\"
        + "\n"
    )
    latex_content += (
        f"\\text{{Главная диагональ }} (b_i) &: {np.array2string(b, separator=', ')} \\\\"
        + "\n"
    )
    latex_content += (
        f"\\text{{Верхняя диагональ }} (c_i) &: {np.array2string(c[:-1], separator=', ')} \\\\"
        + "\n"
    )
    latex_content += (
        f"\\text{{Правые части }} (d_i) &: {np.array2string(d, separator=', ')}" + "\n"
    )
    latex_content += r"\end{align*}" + "\n"

    # Прямой ход
    st.subheader("Шаг 2: Прямой ход (вычисление прогоночных коэффициентов)")
    latex_content += (
        r"\subsection*{Шаг 2: Прямой ход (вычисление прогоночных коэффициентов)}"
    )

    P = np.zeros(n)
    Q = np.zeros(n)
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    st.latex(
        f"P_1 = -\\frac{{c_1}}{{b_1}} = -\\frac{{{c[0]:.4f}}}{{{b[0]:.4f}}} = {P[0]:.4f}"
    )
    st.latex(
        f"Q_1 = \\frac{{d_1}}{{b_1}} = \\frac{{{d[0]:.4f}}}{{{b[0]:.4f}}} = {Q[0]:.4f}"
    )

    latex_content += r"\begin{align*}" + "\n"
    latex_content += (
        f"P_1 &= -\\frac{{c_1}}{{b_1}} = -\\frac{{{c[0]:.4f}}}{{{b[0]:.4f}}} = {P[0]:.4f} \\\\"
        + "\n"
    )
    latex_content += (
        f"Q_1 &= \\frac{{d_1}}{{b_1}} = \\frac{{{d[0]:.4f}}}{{{b[0]:.4f}}} = {Q[0]:.4f}"
        + "\n"
    )
    latex_content += r"\end{align*}" + "\n"

    for i in range(1, n):
        denominator = b[i] + a[i] * P[i - 1]
        if i != n - 1:
            P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator

        st.latex(
            f"P_{{{i+1}}} = -\\frac{{c_{{{i+1}}}}}{{b_{{{i+1}}} + a_{{{i+1}}} P_{{{i}}}}} = "
            f"-\\frac{{{c[i]:.4f}}}{{{b[i]:.4f} + ({a[i]:.4f}) \\cdot ({P[i-1]:.4f})}} = {P[i]:.4f} \\\\"
        )
        st.latex(
            f"Q_{{{i+1}}} = \\frac{{d_{{{i+1}}} - a_{{{i+1}}} Q_{{{i}}}}}{{b_{{{i+1}}} + a_{{{i+1}}} P_{{{i}}}}} = "
            f"\\frac{{{d[i]:.4f} - ({a[i]:.4f}) \\cdot ({Q[i-1]:.4f})}}{{{denominator:.4f}}} = {Q[i]:.4f}"
        )

        latex_content += r"\begin{align*}" + "\n"
        latex_content += (
            f"P_{{{i+1}}} &= -\\frac{{c_{{{i+1}}}}}{{b_{{{i+1}}} + a_{{{i+1}}} P_{{{i}}}}} = "
            f"-\\frac{{{c[i]:.4f}}}{{{b[i]:.4f} + ({a[i]:.4f}) \\cdot ({P[i-1]:.4f})}} = {P[i]:.4f} \\\\"
        ) + "\n"
        latex_content += (
            f"Q_{{{i+1}}} &= \\frac{{d_{{{i+1}}} - a_{{{i+1}}} Q_{{{i}}}}}{{b_{{{i+1}}} + a_{{{i+1}}} P_{{{i}}}}} = "
            f"\\frac{{{d[i]:.4f} - ({a[i]:.4f}) \\cdot ({Q[i-1]:.4f})}}{{{denominator:.4f}}} = {Q[i]:.4f}"
        ) + "\n"
        latex_content += r"\end{align*}" + "\n"

    # Обратный ход
    st.subheader("Шаг 3: Обратный ход (нахождение решения)")
    latex_content += r"\subsection*{Шаг 3: Обратный ход (нахождение решения)}"

    x = np.zeros(n)
    x[-1] = Q[-1]

    st.latex(f"x_{{{n}}} = Q_{{{n}}} = {x[-1]:.4f}")

    latex_content += r"\begin{align*}" + "\n"
    latex_content += f"x_{{{n}}} &= Q_{{{n}}} = {x[-1]:.4f}" + "\n"
    latex_content += r"\end{align*}" + "\n"

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

        st.latex(
            f"x_{{{i+1}}} = P_{{{i+1}}} \\cdot x_{{{i+2}}} + Q_{{{i+1}}} = "
            f"({P[i]:.4f}) \\cdot ({x[i+1]:.4f}) + ({Q[i]:.4f}) = {x[i]:.4f}"
        )

        latex_content += r"\begin{align*}" + "\n"
        latex_content += (
            f"x_{{{i+1}}} &= P_{{{i+1}}} \\cdot x_{{{i+2}}} + Q_{{{i+1}}} = "
            f"({P[i]:.4f}) \\cdot ({x[i+1]:.4f}) + ({Q[i]:.4f}) = {x[i]:.4f}"
        ) + "\n"
        latex_content += r"\end{align*}" + "\n"

    # Итоговое решение
    st.subheader("Итоговое решение:")
    st.latex(f"x={utils.matrix_to_latex(x)}")

    latex_content += r"\subsection*{Итоговое решение:}" + "\n"
    latex_content += f"\[x= {utils.matrix_to_latex(x)} \]\n"

    latex_content += r"\end{document}"
    pdf_bytes = utils.compile_latex_to_pdf(latex_content)
    return pdf_bytes


def solve_iter_and_zeidel(A, b, epsilon, max_iterations=1000):
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath}
\begin{document}
"""

    st.subheader("Исходная матрица и вектор правых частей:")
    st.latex(f"A= {utils.matrix_to_latex(A)}")
    st.latex(f"b= {utils.matrix_to_latex(b)}")

    latex_content += r"\subsection*{Исходная матрица и вектор правых частей:}"
    latex_content += f"\[ A= {utils.matrix_to_latex(A)} \]"
    latex_content += f"\[ b= {utils.matrix_to_latex(b)} \]"

    n = len(b)

    # Метод простых итераций (Якоби)
    st.subheader("Метод простых итераций (Якоби):")
    latex_content += r"\subsection*{Метод простых итераций (Якоби):}"
    x = np.zeros(n)
    for iteration in range(max_iterations):
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = (
                b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1 :], x[i + 1 :])
            ) / A[i, i]

        if np.linalg.norm(x_new - x) < epsilon:
            iter_x = x_new
            iter_amount = iteration + 1
            break
        x = x_new
    else:
        st.write("Достигнуто максимальное число итераций!")
        iter_x = x
        iter_amount = max_iterations
    st.latex(f"x={utils.matrix_to_latex(iter_x)}")
    st.latex(f"\\text{{Количество итераций: }} {iter_amount}")

    latex_content += f"\[ x={utils.matrix_to_latex(iter_x)} \]"
    latex_content += f"\\text{{Количество итераций: }} {iter_amount}"

    # Метод Зейделя
    st.subheader("Метод Зейделя:")
    latex_content += r"\subsection*{Метод Зейделя:}"

    x = np.zeros(n)
    for iteration in range(max_iterations):
        x_prev = x.copy()
        for i in range(n):
            x[i] = (
                b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1 :], x_prev[i + 1 :])
            ) / A[i, i]

        if np.linalg.norm(x - x_prev) < epsilon:
            zeidel_x = x
            zeidel_amount = iteration + 1
            break
    else:
        st.write("Достигнуто максимальное число итераций!")
        zeidel_x = x
        zeidel_amount = max_iterations
    st.latex(f"x={utils.matrix_to_latex(zeidel_x)}")
    st.latex(f"\\text{{Количество итераций: }} {zeidel_amount}")

    latex_content += f"\[ x={utils.matrix_to_latex(zeidel_x)} \]"
    latex_content += f"\\text{{Количество итераций: }} {zeidel_amount}"

    latex_content += r"\end{document}"
    pdf_bytes = utils.compile_latex_to_pdf(latex_content)
    return pdf_bytes


def solve_pivot(a, epsilon):
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath}
\begin{document}
"""

    st.subheader("Исходная матрица:")
    st.latex(f"A= {utils.matrix_to_latex(a)}")

    latex_content += "\section{Исходная симметрическая матрица:}"
    latex_content += f"\[ A= {utils.matrix_to_latex(a)} \]"

    n = a.shape[0]
    eigenvectors = np.eye(n)
    iterations = 0

    # Получаем "истинные" собственные значения для сравнения
    true_eigenvalues = np.linalg.eig(a)[0]
    true_eigenvalues.sort()  # Сортируем для корректного сравнения

    # Списки для хранения ошибок и числа итераций
    errors = []
    iteration_counts = []

    while True:
        # 1. Находим максимальный по модулю недиагональный элемент
        upper_tri = np.triu(a, k=1)
        p, q = np.unravel_index(np.argmax(np.abs(upper_tri)), a.shape)
        max_val = abs(a[p, q])

        # 2. Вычисляем текущие собственные значения и сортируем
        current_eigenvalues = np.diag(a).copy()
        current_eigenvalues.sort()

        # 3. Вычисляем погрешность (средняя квадратичная ошибка)
        error = np.sqrt(np.mean((current_eigenvalues - true_eigenvalues) ** 2))
        errors.append(error)
        iteration_counts.append(iterations)

        st.subheader(f"Итерация {iterations + 1}")
        st.latex(
            f"\\text{{Максимальный внедиагональный элемент: }}( a_{{{p+1},{q+1}}} = {max_val:.6f})"
        )
        st.latex(f"\\text{{Текущая погрешность: }}( {error:.6f} )")
        st.latex(
            f"\\text{{Текущая матрица:\n}}A_{{{iterations + 1}}} = {utils.matrix_to_latex(a)}"
        )

        latex_content += f"\n\\subsection{{Итерация {iterations + 1}}}"
        latex_content += f"\nМаксимальный внедиагональный элемент: \\( a_{{{p+1},{q+1}}} = {max_val:.6f} \\)"
        latex_content += f"\n\nТекущая погрешность: \\( {error:.6f} \\)"
        latex_content += f"\n\nТекущая матрица:\n\\[ A_{{{iterations + 1}}} = {utils.matrix_to_latex(a)} \\]"

        # 4. Проверка на выход по точности
        if max_val < epsilon * np.max(np.abs(np.diag(a))):
            st.subheader("Критерий остановки выполнен")
            st.latex(
                f"\\text{{Максимальный внедиагональный элемент }}( a_{{{p+1},{q+1}}} = {max_val:.6f}) \\text{{ стал меньше заданной точности }}( \\varepsilon = {epsilon})."
            )

            latex_content += r"\subsection{Критерий остановки выполнен}" + "\n"
            latex_content += (
                f"Максимальный внедиагональный элемент \\( a_{{{p+1},{q+1}}} = {max_val:.6f} \\) "
                f"стал меньше заданной точности \\( \\varepsilon = {epsilon} \\).\n\n"
            )
            break

        # 5. Вычисляем угол поворота
        theta = 0.5 * np.arctan2(2 * a[p, q], a[p, p] - a[q, q])

        # 6. Строим матрицу вращения
        rotation = np.eye(n)
        c, s = np.cos(theta), np.sin(theta)
        rotation[p, p] = c
        rotation[q, q] = c
        rotation[p, q] = -s
        rotation[q, p] = s

        # 7. Применяем вращение
        a = rotation.T @ a @ rotation
        eigenvectors = eigenvectors @ rotation
        iterations += 1

    # Добавляем результаты в LaTeX-документ
    eigenvalues = np.diag(a)
    eigenvalues_str = utils.matrix_to_latex(eigenvalues)
    eigenvectors_str = utils.matrix_to_latex(eigenvectors)

    st.subheader("Результаты")
    st.write(f"После {iterations} итераций получены следующие результаты:")
    st.write("Собственные значения:")
    st.latex(f"\\lambda_i = {eigenvalues_str}")
    st.write("Собственные векторы (по столбцам):")
    st.latex(f"V = {eigenvectors_str}")

    latex_content += f"""
    \\section{{Результаты}}
    После {iterations} итераций получены следующие результаты:

    \\subsection{{Собственные значения}}
    \\[ \\lambda_i = {eigenvalues_str} \\]

    \\subsection{{Собственные векторы}}
    Собственные векторы (по столбцам):
    \\[ V = {eigenvectors_str} \\]
    """
    latex_content += r"\end{document}"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iteration_counts,
            y=errors,
        )
    )
    fig.update_layout(
        title="Зависимость погрешности от числа итераций",
        xaxis_title="Количество итераций",
        yaxis_title="Среднеквадратичная погрешность",
    )

    pdf_bytes = utils.compile_latex_to_pdf(latex_content)
    return pdf_bytes, fig


def solve_qr(a, epsilon, max_iter=1000):
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath}
\begin{document}"""

    st.subheader("Исходная матрица:")
    st.latex(f"A= {utils.matrix_to_latex(a)}")

    latex_content += f"\n\\section*{{Исходная матрица:}}\n"
    latex_content += f"\[ A^{{(0)}} = {utils.matrix_to_latex(np.array(a))} \]"

    A = np.array(a, dtype=complex)
    n = A.shape[0]
    assert A.shape[1] == n, "Матрица должна быть квадратной"

    # Приведение к Хессенберговой форме
    A = hessenberg(A)

    st.subheader("Хессенбергова форма (до итераций):")
    st.latex(f"H = {utils.matrix_to_latex(A)}")
    st.subheader("QR-разложение матрицы:")

    latex_content += f"\n\\section*{{Хессенбергова форма (до итераций):}}\n"
    latex_content += f"\[ H = {utils.matrix_to_latex(A)} \]"
    latex_content += "QR-разложение матрицы:\n"

    for k in range(max_iter):
        st.subheader(f"Итерация {k+1}")
        latex_content += f"\n\\section*{{Итерация {k+1}}}\n"

        # 2x2 нижний блок
        if n >= 2:
            T = A[-2:, -2:]
            eigenvals = np.linalg.eigvals(T)
            last_diag = A[-1, -1]
            shift = eigenvals[np.argmin(np.abs(eigenvals - last_diag))]
        else:
            shift = A[-1, -1]

        # QR-разложение (с комплексной поддержкой)
        Q, R = qr(A - shift * np.eye(n))  # QR для сдвинутой матрицы
        A = R @ Q + shift * np.eye(n)  # Сдвиг возвращается обратно

        st.latex(f"Q = {utils.matrix_to_latex(Q)}")
        st.latex(f"R = {utils.matrix_to_latex(R)}")
        st.write("Обновленная матрица:")
        st.latex(f"A^{{(k+1)}} = {utils.matrix_to_latex(A)}")

        latex_content += f"\[ Q = " + utils.matrix_to_latex(Q) + r" \]" + "\n"
        latex_content += f"\[ R = " + utils.matrix_to_latex(R) + r" \]" + "\n\n"
        latex_content += "Обновленная матрица $A^{(k+1)} = RQ$:\n"
        latex_content += f"\\[ A^{{(k+1)}} = {utils.matrix_to_latex(A)} \\]\n\n"

        # Проверка сходимости по норме поддиагональных элементов
        off_diag_norm = np.linalg.norm(np.tril(A, -1))

        st.write(f"Норма поддиагональных элементов: {off_diag_norm:.6e}")
        latex_content += f"Норма поддиагональных элементов: ${off_diag_norm:.6e}$\n"

        if off_diag_norm < epsilon:
            st.latex(f"\\text{{Условие сходимости достигнуто }}(\\epsilon = {epsilon})")
            latex_content += (
                f"\nУсловие сходимости достигнуто ($\\epsilon = {epsilon}$)\n"
            )
            break
    else:
        st.write(f"Не сошлось за {max_iter} итераций")
        latex_content += f"\nНе сошлось за {max_iter} итераций. Норма поддиагонали: {off_diag_norm:.2e}\n"

    eigenvalues = np.diag(A)
    st.subheader("Результат - собственные значения (возможно, комплексные):")
    st.latex(f"x= {utils.matrix_to_latex(eigenvalues)}")

    latex_content += "\nРезультат - собственные значения (на диагонали):\n"
    latex_content += f"\[x= {utils.matrix_to_latex(eigenvalues)} \]"
    latex_content += r"\end{document}"

    pdf_bytes = utils.compile_latex_to_pdf(latex_content)
    return pdf_bytes


def display():
    st.title("Лабораторная работа 1. Методы решения задач линейной алгебры")

    st.header("1.1 LU-разложение")
    use_var_matrix = st.checkbox("Подставить значения варианта", key="check_lu")
    if not use_var_matrix:
        quotient_matrix = st_elements.system_input(key="lu")
    else:
        quotient_matrix = np.array(
            [
                [1, 2, -2, 6, 24],
                [-3, -5, 14, 13, 41],
                [1, 2, -2, -2, 0],
                [-2, -4, 5, 10, 20],
            ],
            dtype=float,
        )
    if st.button("Решить систему", key="lu"):
        with st.expander("Показать решение"):
            pdf_bytes = solve_lu(np.array(quotient_matrix, dtype=float).copy())
        st_elements.download_pdf_button(pdf_bytes, filename="solution_lu.pdf")

    st.header("1.2 Метод прогонки")
    use_var_matrix = st.checkbox("Подставить значения варианта", key="check_tm")
    if not use_var_matrix:
        quotient_matrix = st_elements.system_input(key="tm")
    else:
        quotient_matrix = np.array(
            [
                [-11, -9, 0, 0, 0, -122],
                [5, -15, -2, 0, 0, -48],
                [0, -8, 11, -3, 0, -14],
                [0, 0, 6, -15, 4, -50],
                [0, 0, 0, 3, 6, 42],
            ],
            dtype=float,
        )
    if st.button("Решить систему", key="tm"):
        with st.expander("Показать решение"):
            pdf_bytes = solve_thomas(np.array(quotient_matrix, dtype=float).copy())
        st_elements.download_pdf_button(pdf_bytes, filename="solution_thomas.pdf")

    st.header("1.3 Метод простых итераций и метод Зейделя")
    prec = st.number_input(
        "Введите точность метода (знаков после запятой)",
        min_value=1,
        max_value=7,
        key="prec_zm",
    )
    epsilon = float(f"1e-{prec}")
    use_var_matrix = st.checkbox("Подставить значения варианта", key="check_zm")
    if not use_var_matrix:
        quotient_matrix = st_elements.system_input(key="zm")
    else:
        quotient_matrix = np.array(
            [
                [19, -4, -9, -1, 100],
                [-2, 20, -2, -7, -5],
                [6, -5, -25, 9, 34],
                [0, -3, -9, 12, 69],
            ],
            dtype=float,
        )
    if st.button("Решить систему", key="zm"):
        a = np.array(quotient_matrix, dtype=float).copy()
        A = a[:, :-1]
        b = a[:, -1]
        with st.expander("Показать решение"):
            pdf_bytes = solve_iter_and_zeidel(A, b, epsilon=epsilon)
        st_elements.download_pdf_button(pdf_bytes, filename="solution_zeidel.pdf")

    st.header("1.4 Метод вращений")
    prec = st.number_input(
        "Введите точность метода (знаков после запятой)",
        min_value=1,
        max_value=8,
        key="prec_pm",
    )
    epsilon = float(f"1e-{prec}")
    use_var_matrix = st.checkbox("Подставить значения варианта", key="check_pm")
    if not use_var_matrix:
        quotient_matrix = st_elements.matrix_input(key="pm")
    else:
        quotient_matrix = np.array(
            [[-7, 4, 5], [4, -6, -9], [5, -9, -8]],
            dtype=float,
        )
    if st.button("Решить систему", key="pm"):
        a = np.array(quotient_matrix, dtype=float).copy()
        with st.expander("Показать решение"):
            pdf_bytes, fig = solve_pivot(a, epsilon=epsilon)
            st.plotly_chart(fig)
        st_elements.download_pdf_button(pdf_bytes, filename="solution_pivot.pdf")

    st.header("1.5 Метод QR-разложения матриц")
    prec = st.number_input(
        "Введите точность метода (знаков после запятой)",
        min_value=1,
        max_value=8,
        key="prec_qr",
    )
    epsilon = float(f"1e-{prec}")
    use_var_matrix = st.checkbox("Подставить значения варианта", key="check_qr")
    if not use_var_matrix:
        quotient_matrix = st_elements.matrix_input(key="qr")
    else:
        quotient_matrix = np.array(
            [[3, -7, -1], [-9, -8, 7], [5, 2, 2]],
            dtype=float,
        )
    if st.button("Решить систему", key="qr"):
        a = np.array(quotient_matrix, dtype=float).copy()
        with st.expander("Показать решение"):
            pdf_bytes = solve_qr(a, epsilon=epsilon)
        st_elements.download_pdf_button(pdf_bytes, filename="solution_qr.pdf")


display()
