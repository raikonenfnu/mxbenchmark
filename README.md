# MXBenchmark Docker Setup

This repository provides a Docker-based environment for running MXBenchmark with Triton, Wave, and Aiter frameworks.

## Prerequisites

- Docker installed on your system
- AMD GPU with ROCm support
- Access to `/dev/kfd` and `/dev/dri` devices

## Quick Start

### 1. Build the Docker Image

```sh
docker build --network=host -t mxbenchmark:latest .
```

This will create a Docker image with:

- ROCm 7.0 preview environment
- Triton (shared/triton-gfx950-launch branch)
- Wave framework
- Aiter framework
- rocmProfileData for kernel profiling
- All necessary Python dependencies

### 2. Run the Container

```sh
docker run --name mxbenchmark_container \
  -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --network=host \
  --ipc=host \
  -v $(pwd):/workspace/mxbenchmark \
  mxbenchmark:latest
```

### 3. Run Benchmarks

Inside the container, you can run benchmarks with different backends:

#### Triton Backend

```sh
python /workspace/mxbenchmark/benchmark_mxfp4.py --shape 16384 16384 16384 --backend triton
```

#### ASM Backend

```sh
python /workspace/mxbenchmark/benchmark_mxfp4.py --shape 16384 16384 16384 --backend asm
```

#### Wave Backend

```sh
python /workspace/mxbenchmark/benchmark_mxfp4.py --shape 16384 16384 16384 --backend wave
```

## Directory Structure

The container has the following workspace structure:

```
/workspace/
├── aiter/          # ROCm Aiter framework
├── wave/           # Wave framework
├── triton/         # Triton compiler (gfx950-launch branch)
├── mxbenchmark/    # Your local mxbenchmark repo (volume mounted)
├── rocmProfileData/ # ROCm profiling tools
└── process_rpd.py  # RPD processing script
```

## Environment Variables

The following environment variables are pre-configured for optimal performance:

- `WAVE_CACHE_ON=0`
- `AMD_SERIALIZE_KERNEL=3`
- `TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1`
- `AMDGCN_USE_BUFFER_OPS=1`
- `TRITON_HIP_ASYNC_FAST_SWIZZLE=1`
- `TRITON_HIP_USE_ASYNC_COPY=1`
- `TRITON_HIP_USE_BLOCK_PINGPONG=1`

These are automatically set when the container starts.

## Kernel Profiling with rocmProfileData

The container includes rocmProfileData for detailed kernel-level profiling. The `process_rpd.py` script is available at `/workspace/process_rpd.py` for processing profiling data.

## Development Workflow

1. Make changes to your local mxbenchmark repository
2. Changes are immediately reflected in `/workspace/mxbenchmark` inside the container
3. Run benchmarks to test your changes
4. Profile kernel performance as needed

## Stopping and Restarting

To stop the container:

```sh
docker stop mxbenchmark_container
```

To restart the container:

```sh
docker start -i mxbenchmark_container
```

To remove the container:

```sh
docker rm mxbenchmark_container
```

## Troubleshooting

### GPU Access Issues

If you encounter GPU access issues, ensure:

- Your user is in the `video` group: `sudo usermod -a -G video $USER`
- ROCm drivers are properly installed
- `/dev/kfd` and `/dev/dri` devices exist and have proper permissions

### Container Build Issues

If the build fails, try:

- Checking your internet connection (repositories need to be cloned)
- Ensuring you have enough disk space
- Running with `--no-cache` flag: `docker build --no-cache -t mxbenchmark:latest .`

## Notes

- The benchmark commands use environment variables for performance tuning. These are already set in the container.
- The setup focuses on kernel-level profiling with rocmProfileData rather than end-to-end runs.
- All repositories (except mxbenchmark) are cloned at build time and installed in development mode.
