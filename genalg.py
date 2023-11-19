import ast
import utils as ut
import numpy as np
from random_function import RandomFunctionTree


def initialize(population_size, max_nodes_per_id):
    """
    Generate `population_size` random AST with at most `max_nodes_per_id` nodes
    per return variable.
    """
    rft = RandomFunctionTree()

    population = list()
    for _ in range(population_size):
        population.append(rft(max_nodes_per_id))

    return population


def calc_fitness(population, a, b, c_true):
    """
    For every individual in `population`, compute the fitness based on `a` and
    `b` as input with `c_true` as ideal output.
    """
    v = (np.abs(c_true - c_true.mean()) ** 2).sum()

    fitness = list()

    for tree in population:
        c_pred = np.array(ut.evaluate_tree(tree, a, b), dtype=np.int64).reshape(a.shape)
        fitness.append(ut.score(c_true, c_pred, v))

    return np.array(fitness)


def sort_by_fitness(population, fitness):
    idx_sorted = np.argsort(fitness)
    return [population[idx] for idx in idx_sorted], fitness[idx_sorted]


def offspring_via_mutation(population, nr_offspring):
    """
    Choose (with replacement) `nr_offspring` random individuals from
    `population` and mutate them into offspring.
    """

    if nr_offspring == 0:
        return list()

    population_size = len(population)
    rand_idx = np.random.choice(np.arange(population_size), nr_offspring)

    offspring = list()
    for idx in rand_idx:
        tree = population[idx]
        offspring.append(ut.mutate(tree, copy_tree=True))

    return offspring


def offspring_via_crossover(population, nr_offspring):
    """
    Choose (with replacement) `nr_offspring` random pairs of individuals from
    `population` and produce offspring via crossover.
    """

    if nr_offspring == 0:
        return list()

    population_size = len(population)

    offspring = list()
    for _ in range(nr_offspring):
        idx1, idx2 = np.random.choice(population_size, 2, replace=False)
        parent1 = population[idx1]
        parent2 = population[idx2]
        offspring.append(ut.crossover(parent1, parent2))

    return offspring


def kill_and_repopulate(population, percent_kill, percent_elite, percent_mutation):
    """
    The `percent_elite` percent fittest individuals in `population` survive
    unchanged into the next generation. Of the remaining individuals, the
    most unfit `percent_kill` percent are killed. The empty spots are filled
    with offspring. `percent_mutation` percent of the offspring are produced
    via mutation, the rest via crossover.
    """

    population_size = len(population)

    nr_elite = int(np.ceil(percent_elite * population_size / 100))
    nr_kill = int(np.ceil(percent_kill * population_size / 100))
    nr_mutation = int(np.ceil(percent_mutation * population_size / 100))

    # the elite survive unchanged
    population_new = population[-nr_elite:]

    # the most unfit are not used to produce new offspring
    population = population[nr_kill:]

    # determine how many new individuals come from mutation or crossover
    nr_offspring = population_size - nr_elite
    mutation_0_crossover_1 = np.random.rand(nr_offspring) > (percent_mutation / 100)
    nr_crossover = np.sum(mutation_0_crossover_1)
    nr_mutation = nr_offspring - nr_crossover

    # produce offspring
    offspring_mutation = offspring_via_mutation(population, nr_mutation)
    offspring_crossover = offspring_via_crossover(population, nr_crossover)

    population_new.extend(offspring_mutation)
    population_new.extend(offspring_crossover)

    assert len(population_new) == population_size

    # arrange the offspring randomly
    return population_new


def stats(population, fitness, pm):
    """
    Compute some statistics.
    """
    fitness = np.array(fitness)

    pm["min"].append(np.min(fitness))
    pm["10%"].append(np.quantile(fitness, 0.1))
    pm["mean"].append(np.mean(fitness))
    pm["90%"].append(np.quantile(fitness, 0.9))
    pm["max"].append(np.max(fitness))

    pm["n_Mult"].append(
        [ut.count_nodes(population[idx], ast.Mult) for idx in range(-3, 0)]
    )

    return pm


if __name__ == "__main__":
    pass
