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


def test_crossover():
    rng = np.random.default_rng()
    n = 1_000
    n_ch = 700
    for dim in [2, 3, 4]:
        n_eq = dim**2
        n_mul = dim**3 - 1

        eq_weights = rng.integers(-1, 2, (n, n_eq, n_mul), dtype=DTYPE)
        fitness = np.sort(rng.random(n))

        children = crossover(eq_weights, fitness, n_ch, rng)

        assert children.shape == (n_ch, n_eq, n_mul)


@njit
def fitness(u, v, w, ref_mat, alpha=0.5):
    uv = outer_products(u, v)
    wuv = weighted_sums(w, uv)

    diff = np.abs(wuv - ref_mat).astype(np.float64)
    diff_n = diff.shape[0]
    sum_terms = diff.reshape(diff_n, -1).sum(axis=1)

    sum_mult = np.sum(np.abs(w), axis=1)
    num_mult = np.count_nonzero(sum_mult, axis=1).astype(np.float64)

    term1 = 1 / (sum_terms + 1)
    term2 = 1 / (num_mult + 1)
    score = alpha * term1 + (1 - alpha) * term2
    return score, num_mult, term1


def test_fitness():
    n = 1_000
    for dim in [2, 3, 4]:
        n_eq = dim**2
        n_mul = dim**3 - 1
        n_var = dim**2

        u = np.random.randint(-1, 2, (n, n_mul, n_var), dtype=DTYPE)
        v = np.random.randint(-1, 2, (n, n_mul, n_var), dtype=DTYPE)
        w = np.random.randint(-1, 2, (n, n_eq, n_mul), dtype=DTYPE)

        ref_mat = ref_matrices(dim, DTYPE)

        score, num_mult, sumterms = fitness(u, v, w, ref_mat)

        assert score.shape == (n,)
        assert num_mult.shape == (n,)
        assert sumterms.shape == (n,)


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


def test_mutate():
    rng = np.random.default_rng()
    n = 1_000
    p = 0.1234
    for dim in [2, 3, 4]:
        n_eq = dim**2
        n_mul = dim**3 - 1
        n_var = dim**2

        eq_weights = rng.integers(-1, 2, (n, n_eq, n_mul), dtype=DTYPE)
        eq_weights_mut = mutate(eq_weights, p, rng)

        assert eq_weights_mut.shape == (n, n_eq, n_mul)
        assert eq_weights_mut.min() >= -1
        assert eq_weights_mut.max() <= 1

        u = rng.integers(-1, 2, (n, n_mul, n_var), dtype=DTYPE)
        u_mut = mutate(u, p, rng)

        assert u_mut.shape == (n, n_mul, n_var)
        assert u_mut.min() >= -1
        assert u_mut.max() <= 1


@njit
def outer_products(u, v):
    return u[..., :, None] * v[..., None, :]


def test_outer_products():
    n = 1_000
    for dim in [2, 3, 4]:
        n_mul = dim**3 - 1
        n_var = dim**2

        u = np.random.randint(-1, 2, (n, n_mul, n_var), dtype=DTYPE)
        v = np.random.randint(-1, 2, (n, n_mul, n_var), dtype=DTYPE)

        uv = outer_products(u, v)

        uv_ref = np.zeros((n, n_mul, n_var, n_var), dtype=DTYPE)
        for i in range(n):
            for i_m in range(n_mul):
                uv_ref[i, i_m, :, :] = np.outer(u[i, i_m, :], v[i, i_m, :])

        assert np.allclose(uv, uv_ref)


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


def test_ref_matrices():
    mat_ref = np.zeros((4, 4, 4), dtype=DTYPE)
    # c0 = a0 * b0 + a1 * b2
    mat_ref[0, :, :] = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    # c1 = a0 * b1 + a1 * b3
    mat_ref[1, :, :] = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    # c2 = a2 * b0 + a3 * b2
    mat_ref[2, :, :] = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    # c3 = a2 * b1 + a3 * b3
    mat_ref[3, :, :] = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )

    mat = ref_matrices(2, DTYPE)

    assert np.count_nonzero(mat - mat_ref) == 0


@njit
def sort(score, u, v, w, num_mult, sumterms):
    """Sort by `score` in descending order."""
    idxsort = np.argsort(score)[::-1]
    u = u[idxsort]
    v = v[idxsort]
    w = w[idxsort]
    score = score[idxsort]
    num_mult = num_mult[idxsort]
    sumterms = sumterms[idxsort]
    return u, v, w, score, num_mult, sumterms


def stats(score, num_mult, sumterms, pm):
    """
    Compute some statistics assuming arrays being sorted by descending score.
    """
    pm["score min"].append(np.min(score))
    pm["score 10%"].append(np.quantile(score, 0.1))
    pm["score mean"].append(np.mean(score))
    pm["score 90%"].append(np.quantile(score, 0.9))
    pm["score max"].append(np.max(score))

    # the best individuals' numbers of multiplications
    pm["num_mult"].append(num_mult[:3])

    pm["sumterms"].append([f"{x:.2f}" for x in sumterms[:3]])

    return pm


@njit
def weighted_sums(w, uv):
    return (w[:, :, :, None, None] * uv[:, None, :, :, :]).sum(axis=2)


def test_weighted_sums():
    n = 1_000
    for dim in [2, 3, 4]:
        n_eq = dim**2
        n_mul = dim**3 - 1
        n_var = dim**2

        uv = np.random.randint(-10, 10, (n, n_mul, n_var, n_var), dtype=DTYPE)
        w = np.random.randint(-1, 2, (n, n_eq, n_mul), dtype=DTYPE)

        wuv = weighted_sums(w, uv)

        wuv_ref = np.zeros((n, n_eq, n_var, n_var), dtype=DTYPE)
        for i in range(n):
            for i_e in range(n_eq):
                for i_m in range(n_mul):
                    wuv_ref[i, i_e, :, :] += w[i, i_e, i_m] * uv[i, i_m, :, :]

        assert np.allclose(wuv, wuv_ref)


@njit
def run_generations(
    u,
    v,
    w,
    p_mut,
    ref_mat,
    alpha,
    num_elite,
    num_crossover,
    num_mutation,
    num_selection,
    num_generations,
    rng,
):
    for _ in range(num_generations):
        score, num_mult, sumterms = fitness(u, v, w, ref_mat, alpha)

        u, v, w, score, num_mult, sumterms = sort(score, u, v, w, num_mult, sumterms)

        # the elite survive unchanged
        w_new = w.copy()

        # crossover for most children
        # w_new[num_elite : num_elite + num_crossover] = crossover(
        #     w, score, num_crossover, rng
        # )

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

    score, num_mult, sumterms = fitness(u, v, w, ref_mat, alpha)

    u, v, w, score, num_mult, sumterms = sort(score, u, v, w, num_mult, sumterms)

    return u, v, w, score, num_mult, sumterms


def run(args):
    os.makedirs("results", exist_ok=True)

    seed = np.random.randint(10_000_000, 100_000_000)
    rng = np.random.default_rng(seed)
    print("seed =", seed)

    alpha = 1  # smaller alpha <=> fewer multiplications
    dim = 2
    generations = 1000
    percent_elite = 5  # this percent of fittest individuals survive unchanged
    percent_mutation = 5  # this percent of non-elite children individuals sees mutation
    percent_print = 5
    percent_selection = (
        10  # this percent of children are generated via random selection
    )
    population_size = 10000
    p_mut = 0.1

    print("alpha =", alpha)
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
        "sumterms": list(),
        "num_mult": list(),
        "mean(time_per_generation [s])": [0],
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
    u, v, w, score, num_mult, sumterms = run_generations(
        u=u,
        v=v,
        w=w,
        p_mut=p_mut,
        ref_mat=ref_mat,
        alpha=alpha,
        num_elite=num_elite,
        num_crossover=num_crossover,
        num_mutation=num_mutation,
        num_selection=num_selection,
        num_generations=1,
        rng=rng,
    )

    perf = stats(score, num_mult, sumterms, perf)
    print(pd.DataFrame(perf))

    tic_tot = time.time()
    time_tot = 0
    for i in tqdm(range(1, loops + 1)):
        tic = time.time()

        u, v, w, score, num_mult, sumterms = run_generations(
            u=u,
            v=v,
            w=w,
            p_mut=p_mut,
            ref_mat=ref_mat,
            alpha=alpha,
            num_elite=num_elite,
            num_crossover=num_crossover,
            num_mutation=num_mutation,
            num_selection=num_selection,
            num_generations=num_gen_per_loop,
            rng=rng,
        )

        time_tot += time.time() - tic
        perf["generation"].append(i)
        perf = stats(score, num_mult, sumterms, perf)

        mean_t = time_tot / (i * num_gen_per_loop)
        perf["mean(time_per_generation [s])"].append(mean_t)

        tqdm.write("")
        tqdm.write(pd.DataFrame(perf).to_string())
    print(f"Total runtime: {(time.time() - tic_tot) / 60:.2f} minutes.")


def t_e_s_t():
    # test_crossover()
    test_fitness()
    # test_mutate()
    # test_outer_products()
    # test_ref_matrices()
    # test_weighted_sums()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--noresult", action="store_true")
    args = parser.parse_args()

    run(args)
