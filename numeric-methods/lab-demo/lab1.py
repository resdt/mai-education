import os
import subprocess

import numpy as np
import streamlit as st


@st.fragment
def download_pdf_button(pdf_bytes, filename, label="Скачать PDF"):
    st.download_button(
        label=label,
        icon=":material/download:",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
    )


def compile_latex_to_pdf(latex_str):
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/temp.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory=tmp",
                "-jobname=temp",
                "tmp/temp.tex",
            ]
        )

        with open(f"tmp/temp.pdf", "rb") as f:
            pdf_bytes = f.read()
    except subprocess.CalledProcessError:
        print("Ошибка при компиляции LaTeX. Проверьте лог pdflatex.")

    return pdf_bytes


def format_number(x):
    s = f"{x:.4f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def matrix_to_latex(matrix, line=False):
    if matrix.ndim == 1:
        rows = [format_number(x) for x in matrix]
        body = r" \\ ".join(rows)
        return rf"\begin{{pmatrix}}{body}\end{{pmatrix}}"

    _, ncols = matrix.shape
    rows = [" & ".join(format_number(x) for x in row) for row in matrix]
    body = r" \\ ".join(rows)
    print(body)

    if line and ncols > 1:
        spec = " ".join("c" for _ in range(ncols - 1)) + "|c"
        return rf"\left(\begin{{array}}{{{spec}}}{body}\end{{array}}\right)"
    else:
        return rf"\begin{{pmatrix}}{body}\end{{pmatrix}}"


def lu_decomposition(A, verbose=False):
    n = A.shape[0]
    L = np.eye(n)  # Единичная нижняя треугольная матрица
    U = A.copy().astype(float)  # Копия исходной матрицы
    P = np.eye(n)  # Матрица перестановок

    pivot_amount = 0
    for k in range(n - 1):
        # Выбор главного элемента в столбце k
        max_row = k + np.argmax(np.abs(U[k:, k]))

        # Перестановка строк, если необходимо
        if max_row != k:
            if verbose:
                print(f"Перестановка строк {k+1} и {max_row+1}")

            pivot_amount += 1

            U[[k, max_row]] = U[[max_row, k]]  # Для U

            # Перестановка в уже вычисленной части L
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

            # Обновление матрицы перестановок
            P[[k, max_row]] = P[[max_row, k]]

        # Проверка на вырожденность
        if abs(U[k, k]) < 1e-12:
            raise ValueError(
                f"Матрица вырождена на шаге {k+1} (диагональный элемент {U[k, k]})"
            )

        # Вычисление множителей и обновление подматрицы
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U, pivot_amount


def solve_task(a):
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath}
\begin{document}
\section*{Решение системы}
    """
    n = a.shape[0]
    A = a[:, :-1]
    b = a[:, -1]

    st.subheader("Исходная матрица и вектор b:")
    st.latex(r"A = " + matrix_to_latex(A))
    st.latex(r"b = " + matrix_to_latex(b))
    latex_content += r"\subsection*{Исходная матрица и вектор b:}" + "\n"
    latex_content += r"\[ A = " + matrix_to_latex(A) + r" \]" + "\n"
    latex_content += r"\[ b = " + matrix_to_latex(b) + r" \]" + "\n"
    st.divider()

    # LU-разложение
    st.subheader("LU-разложение")
    try:
        P, L, U, pivot_amount = lu_decomposition(A)
        b_permuted = b @ P
    except np.linalg.LinAlgError:
        st.error("Матрица вырождена, LU-разложение невозможно")
        return None

    st.latex(r"L = " + matrix_to_latex(L))
    st.latex(r"U = " + matrix_to_latex(U))
    latex_content += r"\subsection*{LU-разложение}" + "\n"
    latex_content += r"\[ L = " + matrix_to_latex(L) + r" \]" + "\n"
    latex_content += r"\[ U = " + matrix_to_latex(U) + r" \]" + "\n"
    st.divider()

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
                sum_terms.append(f"{format_number(coeff)} \cdot z_{k+1}")
                sum_value += coeff * z[k]

        z[i] = b_permuted[i] - sum_value
        formula = rf"z_{i+1} = b_{i+1}"
        if sum_terms:
            formula += rf" - ({' + '.join(sum_terms)})"

        st.latex(formula + f" = {format_number(z[i])}")
        latex_content += rf"\[ {formula} = {format_number(z[i])} \]" + "\n"

    st.latex(r"z = " + matrix_to_latex(z))
    latex_content += r"\[ z = " + matrix_to_latex(z) + r" \]" + "\n"
    st.divider()

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
                sum_terms.append(f"{format_number(coeff)} \cdot x_{k+1}")
                sum_value += coeff * x[k]

        x[i] = (z[i] - sum_value) / U[i, i]
        formula = rf"x_{i+1} = \frac{{z_{i+1}"
        if sum_terms:
            formula += rf" - ({' + '.join(sum_terms)})"
        formula += rf"}}{{{format_number(U[i,i])}}}"

        st.latex(formula + f" = {format_number(x[i])}")
        latex_content += rf"\[ {formula} = {format_number(x[i])} \]" + "\n"

    st.latex(r"x = " + matrix_to_latex(x))
    latex_content += r"\[ x = " + matrix_to_latex(x) + r" \]" + "\n"
    st.divider()

    # Дополнительные вычисления (определитель, обратная матрица)
    sign = (-1) ** pivot_amount  # Учет перестановок строк
    det_lu = sign * np.prod(np.diag(U))
    st.subheader("Определитель матрицы")
    st.latex(r"\det(A) = " + format_number(det_lu))
    latex_content += r"\subsection*{Определитель матрицы}" + "\n"
    latex_content += rf"\[ \det(A) = {format_number(det_lu)} \]" + "\n"
    st.divider()

    # 1. Проверка невязки Ax - b
    st.subheader("Проверка невязки")
    latex_content += r"\subsection*{Проверка невязки}" + "\n"

    residual = A @ x - b
    residual_norm = np.linalg.norm(residual)

    st.latex(r"Ax - b = " + matrix_to_latex(residual))
    st.latex(r"\|Ax - b\| = " + f"{residual_norm:.2e}")

    latex_content += r"\[ Ax - b = " + matrix_to_latex(residual) + r" \]" + "\n"
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

        st.latex(r"A^{-1} = " + matrix_to_latex(A_inv))
        st.latex(r"A \cdot A^{-1} = " + matrix_to_latex(I_calculated))
        st.latex(r"\|A \cdot A^{-1} - I\| = " + f"{I_error:.2e}")

        latex_content += r"\[ A^{-1} = " + matrix_to_latex(A_inv) + r" \]" + "\n"
        latex_content += (
            r"\[ A \cdot A^{-1} = " + matrix_to_latex(I_calculated) + r" \]" + "\n"
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

    st.latex(r"\det(A)_{\text{LU}} = " + format_number(det_lu))
    st.latex(r"\det(A)_{\text{numpy}} = " + format_number(det_numpy))
    st.latex(r"Относительная\ ошибка = " + f"{rel_error:.2e}")

    latex_content += rf"\[ \det(A)_{{\text{{LU}}}} = {format_number(det_lu)} \]" + "\n"
    latex_content += (
        rf"\[ \det(A)_{{\text{{numpy}}}} = {format_number(det_numpy)} \]" + "\n"
    )
    latex_content += rf"\[ \text{{Относительная ошибка}} = {rel_error:.2e} \]" + "\n"

    if rel_error < 1e-8:
        st.success("Определитель вычислен верно!")
        latex_content += r"\textbf{Определитель вычислен верно!}" + "\n"
    else:
        st.warning("Возможная ошибка в вычислении определителя!")
        latex_content += r"\textbf{Возможная ошибка в вычислении определителя!}" + "\n"

    latex_content += r"\end{document}"
    pdf_bytes = compile_latex_to_pdf(latex_content)
    return pdf_bytes


def display():
    st.title("Лабораторная работа 1. Методы решения задач линейной алгебры")
    var_amount = st.number_input(
        "Введите количество неизвестных системы", min_value=1, max_value=7
    )
    column_alignment = (
        [1] * (var_amount * 2 + 1) + [15 - var_amount * 2 - 1]
        if var_amount < 7
        else [1] * 15
    )
    columns = st.columns(column_alignment)

    quotient_matrix = [[0 for _ in range(var_amount + 1)] for _ in range(var_amount)]
    for i in range((len(columns) - 1) // 2):
        for j in range((len(columns) - 1) // 2):
            quotient_matrix[i][j] = columns[j * 2].number_input(
                label="Введите коэффициент",
                label_visibility="collapsed",
                value=0,
                key=f"var_input_{i}_{j}",
            )

            var_num = j + 1
            if var_num < var_amount:
                columns[j * 2 + 1].latex(f"x_{var_num}+")
            else:
                columns[j * 2 + 1].latex(f"x_{var_num}=")
                quotient_matrix[i][j + 1] = columns[j * 2 + 2].number_input(
                    label="Введите коэффициент",
                    label_visibility="collapsed",
                    value=0,
                    key=f"free_var_input_{i}_{j}",
                )

    if st.button("Решить систему"):
        with st.expander("Показать решение"):
            pdf_bytes = solve_task(np.array(quotient_matrix, dtype=float).copy())

        download_pdf_button(pdf_bytes, filename="solution.pdf")


display()
