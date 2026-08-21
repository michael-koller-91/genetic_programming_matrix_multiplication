import numpy as np

from main import (
    crossover,
    DTYPE,
    fitness,
    mutate,
    outer_products,
    ref_matrices,
    weighted_sums,
)


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


if __name__ == "__main__":
    test_crossover()
    test_fitness()
    test_mutate()
    test_outer_products()
    test_ref_matrices()
    test_weighted_sums()
    print("all tests passed")
