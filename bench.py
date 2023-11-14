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
            timings.append((time.perf_counter() - t) * 1000)

        time.sleep(1.0)

    return (
        statistics.mean(timings),
        statistics.stdev(timings),
        statistics.quantiles(sorted(timings), n=10)[-1],
    )


def get_pytorch_bench(name):
    from pytorch_benches import PYTORCH_BENCHES

    return PYTORCH_BENCHES[name]()


def get_jax_bench(name):
    from jax_benches import JAX_BENCHES

    return JAX_BENCHES[name]()


def get_all_benches():
    from jax_benches import JAX_BENCHES
    from pytorch_benches import PYTORCH_BENCHES

    return sorted(set(JAX_BENCHES.keys()).intersection(set(PYTORCH_BENCHES.keys())))


def run(bench_name):
    ptmean, ptstd, p_p90 = run_bench(args, get_pytorch_bench(bench_name))
    time.sleep(2.0)
    jaxmean, jaxstd, j_p90 = run_bench(args, get_jax_bench(bench_name))

    print(bench_name)
    print("-" * len(bench_name))
    print(f"pytorch:{ptmean:5.1f}ms ±{ptstd:5.2f}ms p90={p_p90:.1f}ms ")
    print(f"jax:    {jaxmean:5.1f}ms ±{jaxstd:5.2f}ms p90={j_p90:.1f}ms")
    print("")


def main(args):
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
