import streamlit as st


def main():
    pages = [
        st.Page("lab1.py", title="Лабораторная работа 1", icon=":material/science:"),
        st.Page("lab2.py", title="Лабораторная работа 2", icon=":material/science:"),
    ]

    display_pages = st.navigation(pages)
    display_pages.run()


if __name__ == "__main__":
    main()
