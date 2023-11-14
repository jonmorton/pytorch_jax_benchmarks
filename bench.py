import argparse
import statistics
import time

from common import Bench


def run_bench(args, bench: Bench):
    bench.setup(args.batch_size)

    timings = []

    for _ in range(args.cycles):
        for _ in range(args.warmup):
            bench.run()

        for _ in range(args.iters):
            t = time.perf_counter()
            bench.run()
            timings.append(time.perf_counter() - t)

        time.sleep(1.0)

    return statistics.mean(timings), statistics.stdev(timings)


def get_pytorch_bench(name):
    from pytorch_benches import PYTORCH_BENCHES

    return PYTORCH_BENCHES[name]()


def get_jax_bench(name):
    from jax_benches import JAX_BENCHES

    return JAX_BENCHES[name]()


def get_all_benches():
    from jax_benches import JAX_BENCHES
    from pytorch_benches import PYTORCH_BENCHES

    return list(set(JAX_BENCHES.keys()).intersection(set(PYTORCH_BENCHES.keys())))


def run(bench_name):
    ptmean, ptstd = run_bench(args, get_pytorch_bench(bench_name))
    time.sleep(2.0)
    jaxmean, jaxstd = run_bench(args, get_jax_bench(bench_name))

    print(bench_name)
    print(f"  pytorch: {ptmean * 1000:.1f}ms±{ptstd * 1000:.2f}ms")
    print(f"  jax: {jaxmean * 1000:.1f}ms±{jaxstd * 1000:.2f}ms")


def main(args):
    import torch

    # jax uses tf32 by default.
    torch.set_float32_matmul_precision("high")

    import jax

    with jax.default_matmul_precision("tensorfloat32"):
        all_benches = get_all_benches()

        if args.benches == "all":
            benches = all_benches
        else:
            benches = args.benches.split(",")

        for b in benches:
            if b not in all_benches:
                raise Exception(f"Invalid bench name '{b}'")
            time.sleep(2.0)
            run(b)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benches", "-b", type=str, default="all")
    parser.add_argument("--warmup", "-w", default=10)
    parser.add_argument("--iters", "-i", default=100)
    parser.add_argument("--cycles", "-c", default=5)
    parser.add_argument("--batch_size", "-n", default=16)
    return parser


if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    main(args)
