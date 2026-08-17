import numpy as np

DTYPE = np.int32


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


def mutate(arr, p, rng):
    # select random entries
    mask = rng.random(arr.shape) < p

    # map {-1, 0, 1} -> {0, 1, 2}
    v = arr + 1

    bit = rng.integers(0, 2, size=arr.shape, dtype=arr.dtype)  # 0 or 1

    # change value and map back to {-1, 0, 1}
    replacement = ((v + 1 + bit) % 3) - 1  # + 1 to ensure the value is changed
    arr[mask] = replacement[mask]

    return arr


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


def main():
    # test_crossover()
    test_mutate()
    # test_outer_products()
    # test_weighted_sums()


if __name__ == "__main__":
    main()
