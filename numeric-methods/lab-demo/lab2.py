import math

import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go
import streamlit as st

import streamlit_elements as st_elements
import utils


def solve_one_eq(epsilon):
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath}
\begin{document}"""

    # Определяем функцию и её производную
    def f(x):
        return 2**x - x**2 - 0.5

    def df(x):
        return 2**x * math.log(2) - 2 * x

    # Для метода простой итерации нужно определить g(x)
    def g(x):
        return math.sqrt(2**x - 0.5)

    st.subheader("Функция:")
    st.latex("f(x) = 2^x - x^2 - 0.5")

    # Построение графика функции
    x_vals = np.linspace(-1, 3, 400)
    y_vals = [f(x) for x in x_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="f(x)"))
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=[0] * len(x_vals),
            mode="lines",
            name="y = 0",
            line=dict(color="white", dash="dash"),
        )
    )

    fig.update_layout(
        title="График функции для определения начального приближения",
        xaxis_title="x",
        yaxis_title="f(x)",
    )
    st.plotly_chart(fig)

    # Метод Ньютона
    st.subheader("Метод Ньютона")
    latex_content += r"\section*{Метод Ньютона}" + "\n"

    x0 = 1.5  # начальное приближение, можно подобрать графически
    x = x0
    iterations = 0

    st.latex(f"\\text{{Начальное приближение: }}x_0 = {x0:.5f}")

    latex_content += r"Начальное приближение: $x_0 = {:.5f}$".format(x0) + "\n\n"
    latex_content += r"\begin{tabular}{|c|c|c|c|}" + "\n"
    latex_content += r"\hline" + "\n"
    latex_content += r"$n$ & $x_n$ & $f(x_n)$ & $f'(x_n)$ \\" + "\n"
    latex_content += r"\hline" + "\n"

    while True:
        fx = f(x)
        dfx = df(x)
        x_new = x - fx / dfx
        iterations += 1

        st.subheader(f"Итерация {iterations}:")
        st.latex(f"x={x}")
        st.latex(f"f(x)={fx}")
        st.latex(f"f'(x)={dfx}")

        latex_content += (
            "{} & {:.5f} & {:.5f} & {:.5f} \\\\".format(iterations, x, fx, dfx) + "\n"
        )
        latex_content += r"\hline" + "\n"
        if abs(x_new - x) < epsilon:
            break
        x = x_new

    st.subheader("Итог:")
    st.latex(
        f"x \\approx {x_new:.5f} \\text{{ после }} {iterations} \\text{{ итераций при }} \\varepsilon = {epsilon}."
    )

    latex_content += r"\end{tabular}" + "\n\n"
    latex_content += (
        r"Итог: $x \approx {:.5f}$ после {} итераций при $\varepsilon = {}$.".format(
            x_new, iterations, epsilon
        )
        + "\n\n"
    )

    # Метод простой итерации
    st.subheader("Метод простой итерации")
    latex_content += r"\section*{Метод простой итерации}" + "\n"
    x0 = 1.5
    x = x0
    iterations = 0

    st.latex(f"\\text{{Начальное приближение: }}x_0 = {x0:.5f}")

    latex_content += r"Начальное приближение: $x_0 = {:.5f}$".format(x0) + "\n\n"
    latex_content += r"\begin{tabular}{|c|c|}" + "\n"
    latex_content += r"\hline" + "\n"
    latex_content += r"$n$ & $x_n$ \\" + "\n"
    latex_content += r"\hline" + "\n"

    st.subheader("Процесс итерации")
    while True:
        x_new = g(x)
        iterations += 1

        st.latex(f"x_{{{iterations}}}={x}")

        latex_content += "{} & {:.5f} \\\\".format(iterations, x) + "\n"
        latex_content += r"\hline" + "\n"

        if abs(x_new - x) < epsilon:
            break
        x = x_new

    st.subheader("Итог:")
    st.latex(
        f"x \\approx {x_new:.5f} \\text{{ после }} {iterations} \\text{{ итераций при }} \\varepsilon = {epsilon}."
    )

    latex_content += r"\end{tabular}" + "\n\n"
    latex_content += (
        r"Итог: $x \approx {:.5f}$ после {} итераций при $\varepsilon = {}$.".format(
            x_new, iterations, epsilon
        )
        + "\n\n"
    )
    latex_content += r"\end{document}"

    pdf_bytes = utils.compile_latex_to_pdf(latex_content)
    return pdf_bytes


def solve_sys_eq(epsilon):
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian]{babel}
\usepackage{amsmath}
\begin{document}"""

    # Определяем функции системы
    def f1(x1, x2):
        return (x1**2 + 4) * x2 - 8

    def f2(x1, x2):
        return (x1 - 1) ** 2 + (x2 - 1) ** 2 - 4

    # Частные производные для Якобиана
    def df1_dx1(x1, x2):
        return 2 * x1 * x2

    def df1_dx2(x1, x2):
        return x1**2 + 4

    def df2_dx1(x1, x2):
        return 2 * (x1 - 1)

    def df2_dx2(x1, x2):
        return 2 * (x2 - 1)

    # Построение графика для поиска начального приближения
    x1_vals = np.linspace(-2, 4, 400)
    x2_vals = np.linspace(-2, 4, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    F1 = (X1**2 + 4) * X2 - 8
    F2 = (X1 - 1) ** 2 + (X2 - 1) ** 2 - 4
    fig = go.Figure()

    # Линия f1 = 0
    fig.add_trace(
        go.Contour(
            z=F1,
            x=x1_vals,
            y=x2_vals,
            colorscale="Blues",
            showscale=False,  # убираем шкалу справа
            contours=dict(start=0, end=0, size=1, coloring="lines", showlabels=False),
            name=r"f_1(x_1, x_2) = 0",
            line=dict(width=3),
        )
    )

    # Линия f2 = 0
    fig.add_trace(
        go.Contour(
            z=F2,
            x=x1_vals,
            y=x2_vals,
            colorscale="Reds",
            showscale=False,  # убираем шкалу справа
            contours=dict(start=0, end=0, size=1, coloring="lines", showlabels=False),
            name=r"f_2(x_1, x_2) = 0",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title="Графики функций системы",
        xaxis_title="x1",
        yaxis_title="x2",
        template="simple_white",
        legend_title_text="Функции",
        width=800,
        height=600,
    )
    st.plotly_chart(fig)

    # Метод Ньютона
    st.subheader("Метод Ньютона")
    latex_content += r"\section*{Метод Ньютона для системы}" + "\n"

    errors_newton = []

    x1 = 2.0
    x2 = 1.5
    iterations = 0

    st.latex(
        f"\\text{{Начальное приближение: }}x_1^{{(0)}} = {x1:.5f}, x_2^{{(0)}} = {x2:.5f}"
    )
    latex_content += (
        r"Начальное приближение: $x_1^{{(0)}} = {:.5f}$, $x_2^{{(0)}} = {:.5f}$".format(
            x1, x2
        )
        + "\n\n"
    )
    latex_content += r"\begin{tabular}{|c|c|c|c|c|c|}" + "\n"
    latex_content += r"\hline" + "\n"
    latex_content += (
        r"$n$ & $x_1$ & $x_2$ & $f_1(x_1,x_2)$ & $f_2(x_1,x_2)$ & $\|\Delta x\|$ \\"
        + "\n"
    )
    latex_content += r"\hline" + "\n"

    while True:
        f1_val = f1(x1, x2)
        f2_val = f2(x1, x2)

        J = np.array(
            [[df1_dx1(x1, x2), df1_dx2(x1, x2)], [df2_dx1(x1, x2), df2_dx2(x1, x2)]]
        )

        F = np.array([-f1_val, -f2_val])

        delta = np.linalg.lstsq(J, F, rcond=None)[0]

        x1_new = x1 + delta[0]
        x2_new = x2 + delta[1]

        norm = np.linalg.norm(delta)
        errors_newton.append(norm)
        iterations += 1

        st.subheader(f"Итерация {iterations}:")
        st.latex(f"x_1={x1}")
        st.latex(f"x_2={x2}")
        st.latex(f"f_1(x_1, x_2)={f1_val}")
        st.latex(f"f_2(x_1, x_2)={f2_val}")
        st.latex(f"||\\Delta x||={norm}")

        latex_content += (
            "{} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\".format(
                iterations, x1, x2, f1_val, f2_val, norm
            )
            + "\n"
        )
        latex_content += r"\hline" + "\n"

        if norm < epsilon:
            break

        x1, x2 = x1_new, x2_new

    st.subheader("Итог:")
    st.latex(
        f"(x_1, x_2) \\approx ({x1_new:.5f}, {x2_new:.5f}) \\text{{ после }}{iterations} \\text{{ итераций при }}\\varepsilon = {epsilon}."
    )

    latex_content += r"\end{tabular}" + "\n\n"
    latex_content += (
        r"Итог: $(x_1, x_2) \approx ({:.5f}, {:.5f})$ после {} итераций при $\varepsilon = {}$.".format(
            x1_new, x2_new, iterations, epsilon
        )
        + "\n\n"
    )

    # Метод простой итерации
    st.subheader("Метод простой итерации")
    latex_content += r"\section*{Метод простой итерации для системы}" + "\n"

    errors_simple = []

    x1 = 2.0
    x2 = 1.5
    iterations = 0

    st.latex(
        f"\\text{{Начальное приближение: }}x_1^{{(0)}} = {x1:.5f}, x_2^{{(0)}} = {x2:.5f}"
    )
    latex_content += (
        r"Начальное приближение: $x_1^{{(0)}} = {:.5f}$, $x_2^{{(0)}} = {:.5f}$".format(
            x1, x2
        )
        + "\n\n"
    )

    latex_content += r"\begin{tabular}{|c|c|c|c|c|}" + "\n"
    latex_content += r"\hline" + "\n"
    latex_content += r"$n$ & $x_1$ & $x_2$ & $\phi_1(x_2)$ & $\phi_2(x_1)$ \\" + "\n"
    latex_content += r"\hline" + "\n"

    def phi1(x2):
        return 1 + math.sqrt(4 - (x2 - 1) ** 2)

    def phi2(x1):
        return 8 / (x1**2 + 4)

    while True:
        x1_new = phi1(x2)
        x2_new = phi2(x1_new)

        error = max(abs(x1_new - x1), abs(x2_new - x2))
        errors_simple.append(error)

        st.subheader(f"Итерация {iterations + 1}:")
        st.latex(f"x_1={x1}")
        st.latex(f"x_2={x2}")
        st.latex(f"\\phi_1(x_2)={x1_new}")
        st.latex(f"\\phi_2(x_1)={x2_new}")

        latex_content += (
            "{} & {:.5f} & {:.5f} & {:.5f} & {:.5f} \\\\".format(
                iterations, x1, x2, x1_new, x2_new
            )
            + "\n"
        )
        latex_content += r"\hline" + "\n"

        if max(abs(x1_new - x1), abs(x2_new - x2)) < epsilon:
            break

        x1, x2 = x1_new, x2_new
        iterations += 1

    st.subheader("Итог:")
    st.latex(
        f"(x_1, x_2) \\approx ({x1_new:.5f}, {x2_new:.5f}) \\text{{ после }} {iterations} \\text{{ итераций при }} \\varepsilon = {epsilon}."
    )

    latex_content += r"\end{tabular}" + "\n\n"
    latex_content += (
        r"Итог: $(x_1, x_2) \approx ({:.5f}, {:.5f})$ после {} итераций при $\varepsilon = {}$.".format(
            x1_new, x2_new, iterations, epsilon
        )
        + "\n\n"
    )
    latex_content += r"\end{document}"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(errors_newton) + 1)),
            y=errors_newton,
            mode="lines+markers",
            name="Метод Ньютона",
            line=dict(color="royalblue", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(errors_simple) + 1)),
            y=errors_simple,
            mode="lines+markers",
            name="Метод простой итерации",
            line=dict(color="firebrick", width=2),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title="Сравнение сходимости методов: Ньютон и Простая итерация",
        xaxis_title="Номер итерации",
        yaxis_title="Норма погрешности",
        legend_title="Методы",
        template="plotly_white",
    )
    st.plotly_chart(fig)

    pdf_bytes = utils.compile_latex_to_pdf(latex_content)
    return pdf_bytes


def display():
    st.title("Лабораторная работа 2. Метод решения нелинейных уравнений")

    st.header("2.1 Методы простой итерации и Ньютона для уравнения")
    prec = st.number_input(
        "Введите точность метода (знаков после запятой)",
        min_value=1,
        max_value=8,
        key="prec_one_eq",
    )
    epsilon = float(f"1e-{prec}")

    if st.button("Решить систему", key="check_one_eq"):
        with st.expander("Показать решение"):
            pdf_bytes = solve_one_eq(epsilon)
        st_elements.download_pdf_button(pdf_bytes, filename="solution_one_eq.pdf")

    st.header("2.2 Методы простой итерации и Ньютона для системы уравнений")
    prec = st.number_input(
        "Введите точность метода (знаков после запятой)",
        min_value=1,
        max_value=8,
        key="prec_sys_eq",
    )
    epsilon = float(f"1e-{prec}")

    if st.button("Решить систему", key="check_sys_eq"):
        with st.expander("Показать решение"):
            pdf_bytes = solve_sys_eq(epsilon)
        st_elements.download_pdf_button(pdf_bytes, filename="solution_sys_eq.pdf")


display()
