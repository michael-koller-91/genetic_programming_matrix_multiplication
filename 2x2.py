import genalg
import numpy as np
import utils as ut
import pandas as pd


#
# algorithm parameters
#
percent_elite = 1
percent_kill = 20
percent_mutation = 40

print_percent = 5
generations = 100
population_size = 400
max_nodes_per_id = 7

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
}

population = genalg.initialize(
    population_size=population_size, max_nodes_per_id=max_nodes_per_id
)

for i in range(generations):
    fitness = genalg.calc_fitness(population, a, b, c)
    population, fitness = genalg.sort_by_fitness(population, fitness)

    idx_1p0 = np.abs(fitness - 1.0) < 1e-7
    for count, idx in enumerate(idx_1p0):
        if idx is True:
            print(ut.unparse(population[count]))

    if i % int(print_percent / 100 * generations) == 0:
        performance_metrics["generation"].append(i)
        performance_metrics = genalg.stats(population, fitness, performance_metrics)
        print(pd.DataFrame(performance_metrics))

    population = genalg.kill_and_repopulate(
        population,
        percent_kill=percent_kill,
        percent_elite=percent_elite,
        percent_mutation=percent_mutation,
    )
