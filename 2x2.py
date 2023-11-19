import os
import time
import genalg
import numpy as np
import utils as ut
import pandas as pd


filename = f"2x2_{np.random.randint(100_000_000)}.txt"
while os.path.exists(filename):
    filename = f"2x2_{np.random.randint(100_000_000)}.txt"
print("filename:", filename)

#
# algorithm parameters
#
percent_elite = 5
percent_mutation = 5

print_percent = 5
generations = 3000
population_size = 250
max_nodes_per_id = 4

#
# data at which each individual is evaluated to determine its fitness
#
a = np.array([[11, -13], [-5, 7]]).astype("int64")
b = np.array([[-2, 3], [17, -19]]).astype("int64")
c = a @ b

performance_metrics = {
    "generation": list(),
    "min": list(),
    "10%": list(),
    "mean": list(),
    "90%": list(),
    "max": list(),
    "n_Mult": list(),
    "mean(time_per_generation)": list(),
}

with open(filename, "w") as fl:
    fl.write("\n")

# generation 0
population = genalg.initialize(
    population_size=population_size, max_nodes_per_id=max_nodes_per_id
)

fitness = genalg.calc_fitness(population, a, b, c)
population, fitness = genalg.sort_by_fitness(population, fitness)
performance_metrics["mean(time_per_generation)"].append(0)
performance_metrics["generation"].append(0)
performance_metrics = genalg.stats(population, fitness, performance_metrics)
print(pd.DataFrame(performance_metrics))

tic_tot = time.time()
time_tot = 0
appended_to_file = False
for i in range(1, generations + 1):
    tic = time.time()
    population = genalg.next_generation(
        population, percent_elite=percent_elite, percent_mutation=percent_mutation
    )
    fitness = genalg.calc_fitness(population, a, b, c)
    population, fitness = genalg.sort_by_fitness(population, fitness)
    time_tot += time.time() - tic

    # should there be a good result, save its source code
    idx_1p0 = np.abs(fitness - 1.0) < 1e-6
    for count, idx in enumerate(idx_1p0):
        if idx is True:
            ut.append_source(filename, ut.tree_to_source(population[count]))
            appended_to_file = True

    if i % int(print_percent / 100 * generations) == 0:
        performance_metrics["generation"].append(i)
        performance_metrics = genalg.stats(population, fitness, performance_metrics)

        mean_t = time_tot / i
        performance_metrics["mean(time_per_generation)"].append(mean_t)

        print(pd.DataFrame(performance_metrics))
        print(
            f"estimated time remaining: {mean_t * (generations - i) / 60:.2f} minutes"
        )

if appended_to_file:
    print("appended to file", filename)
else:
    os.remove(filename)
print(f"Total runtime: {(time.time() - tic_tot) / 60:.2f} minutes.")
