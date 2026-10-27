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
# PID 1 (e.g. `docker run selin-cpd python script.py`); tini takes PID 1 instead.
ENTRYPOINT ["tini", "--"]

# Create working directory
WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip

# ─── log_diversity_measures (Kovacic & Kabierski) ─────────────────────────────
# Its pyproject pins numpy==2.3.5, but cdrift/evaluation.py uses np.NaN (removed
# in NumPy 2.0) and signal-processing-algorithms requires numpy<2. It is
# installed first so requirements.txt below overrides those pins; the 53
# complexity measures were verified to be identical under numpy 1.26.4 and 2.3.5.
# Its Complexity.py / enriched.py modules live in deps/ and are not packaged.
RUN git clone --recurse-submodules https://git01lab.cs.univie.ac.at/kabierskim94/sample_based_process_comparison.git /opt/sample_based_process_comparison \
    && pip install --no-cache-dir -e /opt/sample_based_process_comparison

# ─── cdrift (Adams et al., 2023) ──────────────────────────────────────────────
# The notebooks import cdrift.evaluation and cdrift.utils.helpers; this repo
# only ships evaluation.py, so the full package is taken from upstream.
RUN git clone https://github.com/cpitsch/cdrift-evaluation /opt/cdrift-evaluation

ENV PYTHONPATH=/opt/cdrift-evaluation:/opt/sample_based_process_comparison/deps

# Install python dependencies (overrides the pins above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter port
EXPOSE 8887

# Launch Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8887", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# docker build -t selin-cpd .
# docker run -it --rm -p 8887:8887 -v "${PWD}:/workspace" selin-cpd
