1. Clone repositories

```sh
cd ~/ && git clone https://github.com/triton-lang/triton.git && cd triton && git checkout shared/triton-gfx950-launch
cd ~/ && git clone https://github.com/iree-org/wave.git && cd wave/
cd ~/ && git clone https://github.com/ROCm/aiter.git && cd aiter/ && git checkout cf29be372d2ecd20102cc22b74a64d75f0c99512 && git submodule sync && git submodule update --init --recursive
cd ~/ && git clone https://github.com/raikonenfnu/mxbenchmark
```

2. Setup docker environment

```sh
docker run --name "$USER"_"torch" -it -d --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --network=host --ipc=host -v "$HOME":"$HOME" --workdir /home/$USER rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha /bin/bash

docker attach "$USER"_"torch"
export HOME=$PWD
apt update
```

3. Install libraries inside docker

```sh
cd ~/triton && pip install -e .
cd ~/aiter && python setup.py develop
cd ~/wave && pip install --no-cache-dir -r requirements-iree-pinned.txt --upgrade && pip install -r requirements.txt -e .
pip install matplotlib
pip install numpy==1.26.0
```

4. Test runs

```sh
WAVE_CACHE_ON=0 AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 python ~/mxbenchmark/benchmark_mxfp4.py --shape 16384 16384 16384 --backend triton

WAVE_CACHE_ON=0 AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 python ~/mxbenchmark/benchmark_mxfp4.py --shape 16384 16384 16384 --backend asm

WAVE_CACHE_ON=0 AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 python ~/mxbenchmark/benchmark_mxfp4.py --shape 16384 16384 16384 --backend wave
```

**NOTE: Above are e2e runs, we are more interested in kernel time, so we need to set up rocmProfileData for that**

5. set up rocmProfileData

```sh
cd ~/ && git clone https://github.com/ROCm/rocmProfileData
cd rocmProfileData
apt-get install sqlite3 libsqlite3-dev && apt-get install libfmt-dev && make; make install
cd ~/ && wget https://gist.githubusercontent.com/raikonenfnu/7d10e109a21a9c337a6f71f9f8a6b3eb/raw/1b0e3a6cf3508c7d555b7a9656c65623911ddc19/process_rpd.py
```
