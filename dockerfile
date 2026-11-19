# ─── Base image ───────────────────────────────────────────────────────────────
# CPU only. Python 3.12 because log_diversity_measures requires Python 3.12+.
FROM python:3.12-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    tini \
    && rm -rf /var/lib/apt/lists/*

# pm4py looks up its parent process on import, which fails when Python runs as
# PID 1; tini takes PID 1 instead.
ENTRYPOINT ["tini", "--"]

# Create working directory
WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip


# ─── cdrift (Adams et al., 2023) ──────────────────────────────────────────────
# The notebooks import cdrift.evaluation and cdrift.utils.helpers; this repo
# only ships evaluation.py, so the full package is taken from upstream.
RUN git clone https://github.com/cpitsch/cdrift-evaluation /opt/cdrift-evaluation

ENV PYTHONPATH=/opt/cdrift-evaluation

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter port
EXPOSE 8887

# Launch Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8887", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# docker build -t cbcdd-image .
# docker run -it --rm -p 8887:8887 -v "${PWD}:/workspace" cbcdd-image
