import argparse
import sys
import os
import torch
import triton
import aiter
from aiter.ops.triton.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffled_scales,
)
from aiter.ops.shuffle import shuffle_weight
from op_tests.triton_tests.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs

TRITON_HIP_PRESHUFFLE_SCALES = (
    os.environ.get("TRITON_HIP_PRESHUFFLE_SCALES", "0") == "1"
)

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)
from iree.turbine.kernel.wave.constraints import (
    ScaledMMAType,
)

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32

def get_mxfp4_gemm(shape, c_dtype):
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4
    c_wave_dtype = torch_dtype_to_wave(c_dtype)
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, vector_shapes={B : 0}, mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def gemm_afp4_wfp4_wave(
        a: tkl.Memory[B, M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[B, M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.bf16],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[B, M, N, tkl.f32]) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        casted = tkw.cast(repeat, c_wave_dtype)
        tkw.write(casted, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 256,
        BLOCK_N: 128,
        BLOCK_K: 256,
        # B: shape[0],
        # M: shape[1],
        N: shape[2],
        K: shape[3],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = [B, M]
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        wave_runtime=False,
        dump_intermediates="./inter",
        dynamic_symbols=dynamic_symbols,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        use_stride_cache_swizzle=True,
        waves_per_eu=1,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm_afp4_wfp4_wave)
    return gemm

def get_x_vals():
    x_vals = [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
    ]
    return x_vals


def run_benchmark(args):
    assert args.shape, "User can specify --shape or --model MODEL -M VAL exclusively"

    x_names = ["B", "M", "N", "K"]
    if args.shape:
        x_vals_list = [args.shape]
    else:
        x_vals_list = get_x_vals()

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = ["TFlops"]
    line_vals = ["triton"]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name="GEMM MXFP4 x MXFP4 Benchmark",
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4_blockscale(B, M, N, K, metric, provider):
        c_dtype = torch.bfloat16
        # x, w, _, _, x_scale, w_scale, _, _ = generate_gemm_afp4wfp4_inputs(
        #     M, N, K, c_dtype
        # )
        x = torch.randn((B * M, K), device="cuda", dtype=c_dtype)
        w = torch.randn((N, K), device="cuda", dtype=c_dtype)
        quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
        _, x_scale = quant_func(x, shuffle=False)
        _, w_scale = quant_func(w, shuffle=False)
        x, x_scales_shuffle = quant_func(x, shuffle=True)
        w, w_scales_shuffle = quant_func(w, shuffle=True)
        wshuffle = shuffle_weight(w, layout=(16, 16))
        # flops
        flops = 2.0 * B * M * N * K
        # memory transfer
        mem_read = x.numel() * x.element_size() + w.numel() * w.element_size()
        mem_read += (
            x_scale.numel() * x_scale.element_size()
            + w_scale.numel() * w_scale.element_size()
        )
        mem_write = (M * N) * 2  # TODO: Fix for c_dtype != bf16
        mem = mem_read + mem_write
        out = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=c_dtype)

        if TRITON_HIP_PRESHUFFLE_SCALES:
            ms = triton.testing.do_bench(
                lambda: gemm_afp4wfp4_preshuffled_scales(
                    x, w, x_scale, w_scale, c_dtype, out
                ),
                warmup=15,
                rep=50,
            )
        else:
            if args.backend == "wave":
                wave_shape = (B, M, N, K)
                gemm = get_mxfp4_gemm(wave_shape, c_dtype)
                wave_out = torch.empty(B, M, N, device=x.device, dtype=c_dtype)
                # gemm(x, x_scale, w_t, w_scale, wave_out)
                # triton_out = torch.empty(M, N, device="cuda", dtype=c_dtype)
                # gemm_afp4wfp4(x, w.T, x_scale, w_scale, c_dtype, triton_out)
                # torch.testing.assert_close(triton_out, wave_out)
                ms = triton.testing.do_bench(
                    lambda: gemm(x.view([B,M,K//2]).view(dtype=torch.uint8), x_scale.view([B,M,K//32]).view(torch.uint8), w, w_scale.view(torch.uint8), wave_out),
                    warmup=25,
                    rep=100,
                )
            elif args.backend == "triton":
                triton_out = torch.empty(B * M, N, device="cuda", dtype=c_dtype)
                ms = triton.testing.do_bench(
                    lambda: gemm_afp4wfp4(x, w.T, x_scale.view(torch.uint8), w_scale.view(torch.uint8), c_dtype, triton_out),
                    warmup=25,
                    rep=100,
                )
            elif args.backend == "ck":
                ck_out = torch.empty((B*M + 255) // 256 * 256, N, device="cuda", dtype=c_dtype)
                ms = triton.testing.do_bench(
                    lambda: aiter.gemm_a4w4_blockscale(x, w, x_scales_shuffle, w_scales_shuffle, ck_out),
                    warmup=25,
                    rep=100,
                )
            elif args.backend == "asm":
                asm_out = torch.empty((B*M + 255) // 256 * 256, N, device="cuda", dtype=c_dtype)
                bias = torch.zeros(M, N, dtype=c_dtype)
                ms = triton.testing.do_bench(
                    lambda: aiter.gemm_a4w4_asm(x, w, x_scales_shuffle, w_scales_shuffle, asm_out, bias, bpreshuffle=False),
                    warmup=25,
                    rep=100,
                )

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "throughput":
            tflops = flops / ms * 1e-9
            return tflops
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_gemm_afp4wfp4_blockscale.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP4 x MXFP4 GEMM",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    parser.add_argument(
        "-M",
        type=int,
        default=4096,
        help="M dim of model benchmark if only one model is under test",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        metavar=("B", "M", "N", "K"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="metric to plot",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["triton", "wave", "ck", "asm"],
        default="triton",
        help="backend to run gemm",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
