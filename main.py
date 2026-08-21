# flake8: noqa E203
from tqdm import tqdm
import datetime as dt
import numba
import numpy as np
import os
import pandas as pd
import time

DTYPE = np.int32
NJIT_ON = False  # set to False to disable numba.njit


def njit(f):
    def decorate(f):
        if NJIT_ON:
            return numba.njit(cache=True)(f)
        else:
            return f

    return decorate(f)


@njit
def count_multiplications(w):
    w_abs_sum = np.sum(np.abs(w), axis=1)
    return np.count_nonzero(w_abs_sum, axis=1).astype(np.float64)


@njit
def crossover(eq_weights, fitness, n_children, rng):
    n_eq = eq_weights.shape[1]

    # --- parent selection (fitness-proportional) ---
    cdf = np.cumsum(fitness)
    cdf /= cdf[-1]  # guarantees cdf[-1] == 1.0
    u = rng.random((n_children, 2))
    parents = np.searchsorted(cdf, u)  # (n_children, 2) parent indices
    # Above 4 lines are equivalent to (but faster than):
    # parents = np.random.choice(
    #     np.arange(len(fitness)), size=(n_children, 2), p=fitness / fitness.sum()
    # )

    # --- cross over C-equations ---
    # 0: choose father | 1: choose mother
    choice = rng.integers(0, 2, size=(n_children, n_eq))
    # map binary choice to parent index
    eq_parents = np.take_along_axis(parents, choice, axis=1)  # (n_children, n_eq)
    # Per row, the following is for example computed:
    #   choice = [0, 1, 1, 1]
    # with
    #   parents = [5, 9]
    # resulting in
    #   eq_parents = [5, 9, 9, 9]

    # single gather: children[l, e, :] = eq_weights[eq_parents[l, e], e, :]
    children = eq_weights[
        eq_parents, np.arange(n_eq)
    ]  # (n_children, n_eq, eq_weights.shape[2])

    return children


@njit
def evaluate(u, v, w, ref_mat, alpha, beta, gamma):
    """Compute fitness and sort the population by descending score.

    Returns the sorted score and losses together with the reordered
    u, v, w arrays.
    """
    score, l_alpha, l_beta, l_gamma = fitness(u, v, w, ref_mat, alpha, beta, gamma)

    score, idx_score = sort(score)
    u = u[idx_score]
    v = v[idx_score]
    w = w[idx_score]
    l_alpha = l_alpha[idx_score]
    l_beta = l_beta[idx_score]
    l_gamma = l_gamma[idx_score]

    return score, l_alpha, l_beta, l_gamma, u, v, w


@njit
def fitness(u, v, w, ref_mat, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Compute the fitness of each individual in the population.

    Three loss terms are combined into a single score (higher is better):
        l_alpha : number of indices where exactly one of wuv / ref_mat is nonzero
        l_beta  : sum of |wuv - ref_mat| / (|ref_mat| + eps) over nonzero ref_mat
        l_gamma : sum of |wuv| over zero ref_mat entries

    Each term is weighted with its respective scalar and the weighted sum l_tot
    of all terms is converted to a goodness value via exp(-l_tot).
    """
    uv = outer_products(u, v)
    wuv = weighted_sums(w, uv)

    wuv_abs = np.abs(wuv).astype(np.float64)
    ref_abs = np.abs(ref_mat).astype(np.float64)
    diff_abs = np.abs(wuv - ref_mat).astype(np.float64)

    # L_alpha: count indices where exactly one of wuv or ref_mat is nonzero
    mismatch = (wuv != 0) != (ref_mat != 0)
    l_alpha = mismatch.reshape(mismatch.shape[0], -1).sum(axis=1).astype(np.float64)

    # L_beta: |wuv - ref_mat| / (|ref_mat| + eps) over nonzero ref_mat indices
    eps = 1e-12
    beta_term = np.where(ref_mat != 0, diff_abs / (ref_abs + eps), 0.0)
    l_beta = beta_term.reshape(beta_term.shape[0], -1).sum(axis=1)

    # L_gamma: |wuv| over zero ref_mat indices
    gamma_term = np.where(ref_mat == 0, wuv_abs, 0.0)
    l_gamma = gamma_term.reshape(gamma_term.shape[0], -1).sum(axis=1)

    # combine into a "higher is better" score (reciprocal form, like the original)
    l_tot = alpha * l_alpha + beta * l_beta + gamma * l_gamma
    score = np.exp(-l_tot)

    return score, l_alpha, l_beta, l_gamma


@njit
def mutate(arr, p, rng):
    mask = rng.random(arr.shape) < p  # select random entries
    v = arr + 1  # map {-1, 0, 1} -> {0, 1, 2}
    bit = rng.integers(0, 2, size=arr.shape)  # , dtype=arr.dtype)  # 0 or 1

    # change value and map back to {-1, 0, 1}
    replacement = ((v + 1 + bit) % 3) - 1  # + 1 to ensure the value is changed
    out = np.where(mask, replacement, arr)

    # enforce: for every (i, j, :) vector, at least one entry is nonzero
    bad = ~np.any(out != 0, axis=-1)  # shape (arr.shape[0], arr.shape[1])
    if np.any(bad):
        i_idx, j_idx = np.where(bad)  # indices for the bad (i, j)
        pos = rng.integers(
            0, arr.shape[-1], size=i_idx.size
        )  # which of the vector slots to activate
        sign = rng.choice(np.array([-1, 1], dtype=out.dtype), size=i_idx.size)
        out[i_idx, j_idx, pos] = sign

    return out


@njit
def outer_products(u, v):
    return u[..., :, None] * v[..., None, :]


def ref_matrices(d, dtype=DTYPE):
    dd = d**2
    mat = np.zeros((dd, dd, dd), dtype=dtype)
    k = -1
    for i in range(d):
        for j in range(d):
            k += 1
            mat[k] = ref_matrix(d, i, j, dtype=dtype)
    return mat


def ref_matrix(d, row, col, dtype=DTYPE):
    """
    Return an (d^2 x d^2) matrix C such that:
        vecA @ C @ vecB == (A @ B)[i, ell]
    where vecA and vecB are vectorizations of A and B.
    """
    assert 0 <= row < d
    assert 0 <= col < d

    C = np.zeros((d**2, d**2), dtype=dtype)
    for k in range(d):
        # flat index of A[i,k]
        u = row * d + k  # A_{row, k}
        v = k * d + col  # B_{k, col}
        C[u, v] = 1
    return C


@njit
def sort(score):
    """Sort by `score` in descending order."""
    idx_score = np.argsort(score)[::-1]
    return score[idx_score], idx_score


def stats(score, l_alpha, l_beta, l_gamma, num_mult, pm):
    """
    Compute some statistics assuming arrays being sorted by descending score.
    """
    pm["score min"].append(f"{np.min(score):.2e}")
    pm["score 10%"].append(f"{np.quantile(score, 0.1):.2e}")
    pm["score mean"].append(f"{np.mean(score):.2e}")
    pm["score 90%"].append(f"{np.quantile(score, 0.9):.2e}")
    pm["score max"].append(f"{np.max(score):.2e}")

    # the best individuals' numbers of multiplications
    pm["num_mult"].append(num_mult[:3])

    pm["l_alpha"].append([f"{x:.2f}" for x in l_alpha[:3]])
    pm["l_beta"].append([f"{x:.2f}" for x in l_beta[:3]])
    pm["l_gamma"].append([f"{x:.2f}" for x in l_gamma[:3]])

    return pm


@njit
def weighted_sums(w, uv):
    return (w[:, :, :, None, None] * uv[:, None, :, :, :]).sum(axis=2)


@njit
def run_generations(
    u,
    v,
    w,
    p_mut,
    # for fitness
    ref_mat,
    alpha,
    beta,
    gamma,
    #
    num_elite,
    num_crossover,
    num_mutation,
    num_selection,
    num_generations,
    rng,
):
    for _ in range(num_generations):
        score, l_alpha, l_beta, l_gamma, u, v, w = evaluate(
            u, v, w, ref_mat, alpha, beta, gamma
        )

        # the elite survive unchanged
        w_new = w.copy()

        # crossover for most children
        w_new[num_elite : num_elite + num_crossover] = crossover(
            w, score, num_crossover, rng
        )

        # random selection for the rest
        idx = np.arange(u.shape[0])
        rng.shuffle(idx)
        w_new[num_elite + num_crossover :] = w[idx[:num_selection]]

        w = w_new.copy()

        # mutate some of the children
        idx = np.arange(num_elite, u.shape[0])  # don't mutate the elite
        rng.shuffle(idx)
        u[idx[:num_mutation]] = mutate(u[idx[:num_mutation]], p_mut, rng)
        v[idx[:num_mutation]] = mutate(v[idx[:num_mutation]], p_mut, rng)
        w[idx[:num_mutation]] = mutate(w[idx[:num_mutation]], p_mut, rng)

    score, l_alpha, l_beta, l_gamma, u, v, w = evaluate(
        u, v, w, ref_mat, alpha, beta, gamma
    )

    return u, v, w, score, l_alpha, l_beta, l_gamma


def run(args):
    os.makedirs("results", exist_ok=True)

    seed = np.random.randint(10_000_000, 100_000_000)
    rng = np.random.default_rng(seed)
    print("seed =", seed)

    alpha = 1  # smaller alpha <=>
    beta = 1  # smaller alpha <=>
    gamma = 1  # smaller alpha <=>

    dim = 2
    generations = 1000
    percent_elite = 3  # this percent of fittest individuals survive unchanged
    percent_mutation = 5  # this percent of non-elite children individuals sees mutation
    percent_print = 5
    percent_selection = (
        10  # this percent of children are generated via random selection
    )
    population_size = 10000
    p_mut = 0.1

    print("alpha =", alpha)
    print("beta =", beta)
    print("gamma =", gamma)
    print("dim =", dim)
    print("generations =", generations)
    print("percent_elite =", percent_elite)
    print("percent_mutation =", percent_mutation)
    print("percent_print =", percent_print)
    print("percent_selection =", percent_selection)
    print("population_size =", population_size)
    print("p_mut =", p_mut)

    date = dt.datetime.strftime(dt.datetime.now(), "%Y-%M-%d_%Hh%Mm%Ss")
    filename = os.path.join("results", date + ".txt")
    print("filename:", filename)

    if not args.noresult:
        with open(filename, "w") as f:
            f.write(f"alpha = {alpha}")
            f.write(f"beta = {beta}")
            f.write(f"gamma = {gamma}")
            f.write(f"dim = {dim}")
            f.write(f"generations = {generations}")
            f.write(f"percent_elite = {percent_elite}")
            f.write(f"percent_mutation = {percent_mutation}")
            f.write(f"percent_print = {percent_print}")
            f.write(f"percent_selection = {percent_selection}")
            f.write(f"population_size = {population_size}")
            f.write(f"p_mut = {p_mut}")

    perf = {
        "generation": [0],
        "score min": list(),
        "score 10%": list(),
        "score mean": list(),
        "score 90%": list(),
        "score max": list(),
        "l_alpha": list(),
        "l_beta": list(),
        "l_gamma": list(),
        "num_mult": list(),
        "mean(time/gen [s])": [0],
    }

    num_elite = int(np.ceil(population_size * percent_elite / 100))
    num_mutation = int(np.ceil(population_size * percent_mutation) / 100)
    num_selection = int(np.ceil(population_size * percent_selection / 100))
    num_crossover = population_size - num_elite - num_selection
    ref_mat = ref_matrices(dim, dtype=DTYPE)

    num_gen_per_loop = int(np.ceil(percent_print / 100 * generations))
    print("num_gen_per_loop =", num_gen_per_loop)
    loops = int(np.ceil(generations / num_gen_per_loop))

    u = rng.integers(-1, 2, (population_size, dim**3 - 1, dim**2), dtype=DTYPE)
    v = rng.integers(-1, 2, (population_size, dim**3 - 1, dim**2), dtype=DTYPE)
    w = rng.integers(-1, 2, (population_size, dim**2, dim**3 - 1), dtype=DTYPE)

    # numba compile
    u, v, w, score, l_alpha, l_beta, l_gamma = run_generations(
        u=u,
        v=v,
        w=w,
        p_mut=p_mut,
        ref_mat=ref_mat,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        num_elite=num_elite,
        num_crossover=num_crossover,
        num_mutation=num_mutation,
        num_selection=num_selection,
        num_generations=1,
        rng=rng,
    )
    num_mult = count_multiplications(w)

    perf = stats(score, l_alpha, l_beta, l_gamma, num_mult, perf)
    print(pd.DataFrame(perf))

    tic_tot = time.time()
    time_tot = 0
    for i in tqdm(range(1, loops + 1)):
        tic = time.time()

        u, v, w, score, l_alpha, l_beta, l_gamma = run_generations(
            u=u,
            v=v,
            w=w,
            p_mut=p_mut,
            ref_mat=ref_mat,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            num_elite=num_elite,
            num_crossover=num_crossover,
            num_mutation=num_mutation,
            num_selection=num_selection,
            num_generations=num_gen_per_loop,
            rng=rng,
        )
        num_mult = count_multiplications(w)

        time_tot += time.time() - tic
        perf["generation"].append(i)
        perf = stats(score, l_alpha, l_beta, l_gamma, num_mult, perf)

        mean_t = time_tot / (i * num_gen_per_loop)
        perf["mean(time/gen [s])"].append(mean_t)

        tqdm.write("")
        tqdm.write(pd.DataFrame(perf).to_string())
    print(f"Total runtime: {(time.time() - tic_tot) / 60:.2f} minutes.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--noresult", action="store_true")
    args = parser.parse_args()

    run(args)
