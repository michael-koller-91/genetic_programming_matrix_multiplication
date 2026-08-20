from tqdm import tqdm
import datetime as dt
import genetics
import numpy as np
import os
import pandas as pd
import time


def run(args):
    os.makedirs("results", exist_ok=True)

    seed = np.random.randint(10_000_000, 100_000_000)
    np.random.default_rng(seed)
    print("seed =", seed)

    beta = 2**-2
    dim = 2
    generations = 1000
    num_mult = dim**3 - 1
    num_vars = dim**2
    percent_elite = 5  # 100 - percent_elite are generated through breeding
    percent_mutation = 10  # chance that any offspring sees a mutation after breeding
    population_size = 500
    print_percent = 5

    print("beta =", beta)
    print("dim =", dim)
    print("generations =", generations)
    print("num_mult =", num_mult)
    print("num_vars =", num_vars)
    print("percent_elite =", percent_elite)
    print("percent_mutation =", percent_mutation)
    print("population_size =", population_size)
    print("print_percent =", print_percent)

    date = dt.datetime.strftime(dt.datetime.now(), "%Y-%M-%d_%Hh%Mm%Ss")
    filename = os.path.join("results", date + ".txt")
    print("filename:", filename)

    if not args.noresult:
        with open(filename, "w") as f:
            f.write(f"dim: {dim}")
            f.write(f"num_mult: {num_mult}")
            f.write(f"num_vars: {num_vars}")
            f.write(f"population_size: {population_size}")
            f.write(f"seed: {seed}")

    perf_metrics = {
        "generation": list(),
        "score min": list(),
        "score 10%": list(),
        "score mean": list(),
        "score 90%": list(),
        "score max": list(),
        "sum_fitness": list(),
        "num_mult": list(),
        "mean(time_per_generation [s])": list(),
    }

    cref = genetics.gen_mat_equations(dim)[1]

    population = genetics.population_init(population_size, num_mult, num_vars)
    _, sum_fitness, num_mults, scores = genetics.population_fitness(
        population, cref, beta
    )

    scores, population, num_mults, idx_sorted = genetics.sort_by(
        scores, population, num_mults
    )
    sum_fitness = sum_fitness[idx_sorted]
    perf_metrics["mean(time_per_generation [s])"].append(0)
    perf_metrics["generation"].append(0)
    perf_metrics = genetics.stats(scores, num_mults, sum_fitness, perf_metrics)
    print(pd.DataFrame(perf_metrics))

    tic_tot = time.time()
    time_tot = 0
    appended_to_file = False
    for i in tqdm(range(1, generations + 1)):
        tic = time.time()
        population = genetics.next_gen(
            population,
            percent_elite=percent_elite,
            percent_mutation=percent_mutation,
            num_vars=num_vars,
            num_mult=num_mult,
        )
        # t0 = time.time()
        _, sum_fitness, num_mults, scores = genetics.population_fitness(
            population, cref, beta
        )
        # t1 = time.time()
        # tqdm.write(f"population_fitness(): {t1 - t0:.2f}")
        scores, population, num_mults, idx_sorted = genetics.sort_by(
            scores, population, num_mults
        )
        sum_fitness = sum_fitness[idx_sorted]
        time_tot += time.time() - tic

        if i % int(np.ceil(print_percent / 100 * generations)) == 0:
            perf_metrics["generation"].append(i)
            perf_metrics = genetics.stats(scores, num_mults, sum_fitness, perf_metrics)

            mean_t = time_tot / i
            perf_metrics["mean(time_per_generation [s])"].append(mean_t)

            print(pd.DataFrame(perf_metrics))
            print(
                f"estimated time remaining: {mean_t * (generations - i) / 60:.2f} minutes"
            )
    print(f"Total runtime: {(time.time() - tic_tot) / 60:.2f} minutes.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--noresult", action="store_true")
    args = parser.parse_args()

    run(args)
