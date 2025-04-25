import os
import subprocess


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
