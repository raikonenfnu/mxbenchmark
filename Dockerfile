FROM rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha

# Set environment variables for optimal performance
ENV WAVE_CACHE_ON=0 \
    AMD_SERIALIZE_KERNEL=3 \
    TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 \
    AMDGCN_USE_BUFFER_OPS=1 \
    TRITON_HIP_ASYNC_FAST_SWIZZLE=1 \
    TRITON_HIP_USE_ASYNC_COPY=1 \
    TRITON_HIP_USE_BLOCK_PINGPONG=1

# Create workspace directory
WORKDIR /workspace

# Fix broken dependencies and install necessary packages
# Allow apt update to continue even if some repos fail (404 errors from AMD repos)
RUN (apt update || true) && \
    apt --fix-broken install -y && \
    apt install -y \
    git \
    sqlite3 \
    libsqlite3-dev \
    libfmt-dev \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone and setup triton
# RUN git clone https://github.com/triton-lang/triton.git /workspace/triton && \
#     cd /workspace/triton && \
#     pip install -e .
RUN pip install triton

# Clone and setup aiter
RUN git clone https://github.com/ROCm/aiter.git /workspace/aiter && \
    cd /workspace/aiter && \
    git checkout cf29be372d2ecd20102cc22b74a64d75f0c99512 && \
    git submodule sync && \
    git submodule update --init --recursive && \
    python setup.py develop

# Clone and setup wave
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

RUN git clone https://github.com/iree-org/wave.git /workspace/wave && \
    cd /workspace/wave && \
    pip install --no-cache-dir -r requirements-iree-pinned.txt --upgrade && \
    pip install -e .

# Install additional Python dependencies
RUN pip install matplotlib numpy==1.26.0

# Setup rocmProfileData
RUN git clone https://github.com/ROCm/rocmProfileData /workspace/rocmProfileData && \
    cd /workspace/rocmProfileData && \
    make && make install && \
    wget -O /workspace/process_rpd.py https://gist.githubusercontent.com/raikonenfnu/7d10e109a21a9c337a6f71f9f8a6b3eb/raw/1b0e3a6cf3508c7d555b7a9656c65623911ddc19/process_rpd.py

# Create mxbenchmark directory (will be mounted from host)
RUN mkdir -p /workspace/mxbenchmark

# Set working directory to workspace
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
