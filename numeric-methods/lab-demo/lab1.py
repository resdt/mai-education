import streamlit as st


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

    print("=" * 60)
    print("\n".join("\t".join(map(str, row)) for row in quotient_matrix))


display()
