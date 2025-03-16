import subprocess

import streamlit as st
import numpy as np


@st.fragment
def download_pdf_button(pdf_bytes, filename, label="Скачать PDF"):
    st.download_button(label=label,
                       icon=":material/download:",
                       data=pdf_bytes,
                       file_name=filename,
                       mime="application/pdf")


def compile_latex_to_pdf(latex_str):
    with open("temp.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)

    try:
        subprocess.run([
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-jobname=temp",
            "temp.tex"
        ])

        with open(f"temp.pdf", "rb") as f:
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

    if line and ncols > 1:
        spec = " ".join("c" for _ in range(ncols - 1)) + "|c"
        return rf"\left(\begin{{array}}{{{spec}}}{body}\end{{array}}\right)"
    else:
        return rf"\begin{{pmatrix}}{body}\end{{pmatrix}}"


def gauss_sequential(a):
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\begin{document}
\section*{Решение системы}
    """
    a_orig = a.copy()

    st.subheader("Исходная матрица:")
    latex_content += r"\subsection*{Исходная матрица:}" + "\n"
    st.latex(matrix_to_latex(a, line=True))
    latex_content += r"\[ " + matrix_to_latex(a, line=True) + r"\]" + "\n"
    st.divider()

    n = a.shape[0]

    # Прямой ход метода Гаусса
    for k in range(n - 1):
        # Выбор опорного элемента: ищем максимальный по модулю элемент в столбце k начиная со строки k
        max_index = np.argmax(np.abs(a[k:, k])) + k

        if np.abs(a[max_index, k]) < 1e-12:
            st.error("Система вырожденная или имеет бесконечное множество решений")
            latex_content += r"\end{document}"
            return latex_content

        st.subheader(f"Шаг {k+1}: выбор опорного элемента для столбца {k+1} -> строка {max_index+1}")
        latex_content += rf"\subsection*{{Шаг {k+1}: выбор опорного элемента для столбца {k+1} -> строка {max_index+1}}}" + "\n"

        # Если необходимо, меняем местами текущую строку k и строку с максимальным элементом
        if max_index != k:
            a[[k, max_index]] = a[[max_index, k]]
            st.write(f"Перестановка строк {k+1} и {max_index+1}:")
            latex_content += f"Перестановка строк {k+1} и {max_index+1}:\\\\\n"
            st.latex(matrix_to_latex(a, line=True))
            latex_content += r"\[" + matrix_to_latex(a, line=True) + r"\]" + "\n"

        # Нормируем текущую строку
        pivot = a[k, k]

        # Прямой ход: зануляем элементы под опорным
        for i in range(k+1, n):
            factor = a[i, k] / pivot
            if factor != 0:
                a[i, k:] = a[i, k:] - factor * a[k, k:]
                st.write(f"Элиминация в строке {i+1} с множителем {format_number(factor)}:")
                latex_content += f"Элиминация в строке {i+1} с множителем {format_number(factor)}:\\\\\n"
                st.latex(matrix_to_latex(a, line=True))
                latex_content += r"\[" + matrix_to_latex(a, line=True) + r"\]" + "\n"

        st.divider()

    st.subheader("Решение системы")
    latex_content += r"\subsection*{Решение системы}" + "\n"

    # Обратный ход: обратная подстановка
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (a[i, -1] - np.dot(a[i, i+1:n], x[i+1:])) / a[i, i]

    columns = st.columns([1, 9])
    for i in range(n):
        columns[0].latex(f"x_{i+1}={format_number(x[i])}")
        latex_content += rf"\[ x_{{{i+1}}} = {format_number(x[i])} \]" + "\n"

    st.divider()

    st.subheader("Проверка")
    latex_content += r"\subsection*{Проверка}" + "\n"
    st.write("Исходная матрица, столбец свободных членов и решение задачи:")
    latex_content += "Исходная матрица, столбец свободных членов и решение задачи:\\\\\n"
    A = a_orig[:, :-1]
    b = a_orig[:, -1]
    columns = st.columns([1, 1, 1])
    columns[0].latex(r"A = " + matrix_to_latex(A))
    columns[1].latex(r"b = " + matrix_to_latex(b))
    columns[2].latex(r"x = " + matrix_to_latex(x))

    # Вычисляем A*x
    st.write("Вычисляем произведение исходной матрицы на вектор ответа:")
    latex_content += "Вычисляем произведение исходной матрицы на вектор ответа:\\\\\n"
    Ax = A.dot(x)
    columns = st.columns([1, 9])
    columns[0].latex(r"Ax = " + matrix_to_latex(Ax))
    latex_content += r"\[ Ax = " + matrix_to_latex(Ax) + r"\]" + "\n"

    # Вычисляем вектор невязок (разность b - Ax)
    st.write("Вычисляем разность между исходным вектором свободных членов и вектором Ax:")
    latex_content += "Вычисляем разность между исходным вектором свободных членов и вектором Ax:\\\\\n"
    residual_vector = b - Ax
    columns = st.columns([1, 9])
    columns[0].latex(r"b - Ax = " + matrix_to_latex(residual_vector))
    latex_content += r"\[ b - Ax = " + matrix_to_latex(residual_vector) + r"\]" + "\n"

    # Норма невязок
    st.write("Вычисляем норму невязок:")
    latex_content += "Вычисляем норму невязок:\\\\\n"
    residual_norm = np.linalg.norm(residual_vector)
    tolerance = 1e-6
    st.latex(r"\|b - Ax\| = " + f"{residual_norm:.2e}")
    latex_content += rf"\[ \|b - Ax\| = {residual_norm:.2e} \]" + "\n"

    if residual_norm < tolerance:
        st.write("Решение системы проверено: невязка очень мала, система решена верно.")
        latex_content += "Решение системы проверено: невязка очень мала, система решена верно.\\\\\n"
    else:
        st.write("Обнаружена значительная невязка. Проверьте вычисления!")
        latex_content += "Обнаружена значительная невязка. Проверьте вычисления!\\\\\n"

    latex_content += r"\end{document}"

    pdf_bytes = compile_latex_to_pdf(latex_content)
    return pdf_bytes


def display():
    st.title("Лабораторная работа 1. Методы решения задач линейной алгебры")
    var_amount = st.number_input("Введите количество неизвестных системы", min_value=1, max_value=7)
    column_alignment = [1]*(var_amount*2 + 1) + [15 - var_amount*2 - 1] if var_amount < 7 else [1] * 15
    columns = st.columns(column_alignment)

    quotient_matrix = [[0 for _ in range(var_amount + 1)] for _ in range(var_amount)]
    for i in range((len(columns)-1) // 2):
        for j in range((len(columns)-1) // 2):
            quotient_matrix[i][j] = columns[j*2].number_input(label="Введите коэффициент", label_visibility="collapsed", value=0, key=f"var_input_{i}_{j}")

            var_num = j + 1
            if var_num < var_amount:
                columns[j*2+1].latex(f"x_{var_num}+")
            else:
                columns[j*2+1].latex(f"x_{var_num}=")
                quotient_matrix[i][j+1] = columns[j*2+2].number_input(label="Введите коэффициент", label_visibility="collapsed", value=0, key=f"free_var_input_{i}_{j}")

    if st.button("Решить систему"):
        with st.expander("Показать решение"):
            pdf_bytes = gauss_sequential(np.array(quotient_matrix, dtype=float).copy())

        download_pdf_button(pdf_bytes, filename="solution.pdf")


display()
