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


def system_input(key):
    var_amount = st.number_input(
        "Введите количество неизвестных системы",
        min_value=1,
        max_value=7,
        key=f"{key}_var_amount",
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
                key=f"{key}_var_input_{i}_{j}",
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
                    key=f"{key}_free_var_input_{i}_{j}",
                )
    return quotient_matrix


def matrix_input(key):
    var_amount = st.number_input(
        "Введите размерность матрицы",
        min_value=1,
        max_value=7,
        key=f"{key}_var_amount",
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
                key=f"{key}_var_input_{i}_{j}",
            )
    return quotient_matrix
