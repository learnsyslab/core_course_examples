# use the official Python image from Docker Hub (slim in local, deepnote/python in deepnote)
#FROM python:3.8-slim
FROM deepnote/python:3.10

# install system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    build-essential \
    git \
    cmake \
    make \
    gcc \
    g++ \
    ffmpeg \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libcdd-dev \
    python3-dev \
    # for Rust
    pkg-config \
    libssl-dev \
    clang \
    llvm \
    && rm -rf /var/lib/apt/lists/*

# Install cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
# make it more stable for acados build
ENV CARGO_BUILD_JOBS=2
ENV RUSTFLAGS="-C codegen-units=1"

# check
RUN cargo --version

# clone acados
RUN git clone https://github.com/acados/acados.git /acados && \
    cd /acados && \
    git submodule update --recursive --init

# build acados
RUN cd /acados && \
    mkdir build && \
    cd build && \
    cmake .. -DACADOS_WITH_C_INTERFACE=ON -DACADOS_INSTALL_PYTHON=ON && \
    make install

# double check acados_template
RUN cd /acados/interfaces/acados_template && \
    pip install .

# ---- Patch to remove future_fstrings encoding header ----
RUN python - <<'PY'
import re, importlib.util, pathlib
pat = re.compile(r'^\s*#\s*-\*-\s*coding:\s*future_fstrings\s*-\*-\s*$', re.I)
def patch_tree(root: pathlib.Path):
    if not root.exists(): return 0
    n=0
    for p in root.rglob("*.py"):
        try:
            L = p.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        except Exception: continue
        ch=False
        for i in (0,1):
            if i < len(L) and pat.match(L[i]):
                L[i] = "# -*- coding: utf-8 -*-\n"; ch=True
        if ch:
            p.write_text("".join(L), encoding="utf-8"); n+=1
    print("patched", n, "files under", root)
spec = importlib.util.find_spec("acados_template")
if spec and spec.origin:
    patch_tree(pathlib.Path(spec.origin).parent)
patch_tree(pathlib.Path("/acados"))
PY
# --------------------------------------------------

# Install t_renderer manually and replace the deafult one with wrong version
RUN git clone https://github.com/acados/tera_renderer.git /tera_renderer && \
    cd /tera_renderer && \
    cargo update -w && \
    cargo build --release && \
    mkdir -p /acados/bin && \
    install -m 0755 /tera_renderer/target/release/t_renderer /acados/bin/t_renderer

# environment variables
ENV ACADOS_SOURCE_DIR=/acados
ENV ACADOS_INSTALL_DIR=/acados
ENV LD_LIBRARY_PATH=/acados/lib
ENV PYTHONPATH=/acados/interfaces/acados_template/python

# set the working directory
WORKDIR /app

# copy the current directory contents into the container at /app
COPY . /app

# install python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir seaborn cvxopt pycddlib==2.1.0 pytope osqp

# install torch (cpu version)
RUN pip install --no-cache-dir torch==2.3.0+cpu torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# expose the port for jupyterlab
EXPOSE 8888

# entrypoint (bash in local, jupyter in Deepnote)
#CMD ["bash"]
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

