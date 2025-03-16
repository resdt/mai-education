# Use Python 3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-lang-cyrillic \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Install dependencies
RUN uv sync --frozen --no-dev

# Expose the port the app will use
EXPOSE 8501

# Define the command to run the application
CMD ["uv", "run", "streamlit", "run", "main.py"]
