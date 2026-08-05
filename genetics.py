# from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
# from functools import partial
from sympy import simplify
import numpy as np

ADD_SUB = ["+", "-"]


def breed(population, selection_probabilities, n_offspring):
    population_size = len(population)
    offspring = list()
    idx_dad = np.random.choice(population_size, n_offspring, p=selection_probabilities)
    idx_mom = np.random.choice(population_size, n_offspring, p=selection_probabilities)
    for idx in range(n_offspring):
        dad = population[idx_dad[idx]]
        mom = population[idx_mom[idx]]
        c_child = cross_c(dad["c"], mom["c"])
        m_child = cross_m(dad["m"], mom["m"])
        offspring.append({"c": c_child, "m": m_child})
    return offspring


def change_var(var, dim):
    vars = [f"{var[0]}{i}" for i in range(dim)]
    opt1, opt2 = choose(vars, 2)
    if opt1 == var:
        return opt2
    else:
        return opt1


def change_op(op):
    if op == "+":
        return "-"
    else:
        return "+"


def choose(*args, **kwargs):
    return np.random.choice(*args, **kwargs, replace=False)


def count_mult(c, num_m_vars):
    """
    Count how many different m-variables occur in all c-equations.
    """
    ms = np.zeros(num_m_vars, dtype=bool)
    for ci in c:
        for m in ci[1][::2]:
            i = int(m[1:])
            ms[i] = True
    return np.sum(ms)


def cross_c(c_dad, c_mom):
    """
    Choose half of the c-equations from dad and the other half from mom.
    """
    indices = np.ones(len(c_dad), dtype=bool)
    falses = choose(np.arange(len(c_dad)), len(c_dad) // 2)
    indices[falses] = False
    c_child = list()
    for i in range(len(c_dad)):
        if indices[i]:
            c_child.append(c_dad[i])
        else:
            c_child.append(c_mom[i])
    return c_child


def cross_m(m_dad, m_mom):
    """
    Choose half of the m-equations from dad and the other half from mom.
    """
    indices = np.ones(len(m_dad), dtype=bool)
    falses = choose(np.arange(len(m_dad)), len(m_dad) // 2)
    indices[falses] = False
    m_child = list()
    for i in range(len(m_dad)):
        var = f"m{i}"
        if indices[i]:
            m_child.append([var, *m_dad[i][1:]])
        else:
            m_child.append([var, *m_mom[i][1:]])
    return m_child


def evaluate(func, *args):
    comp = compile(func, "", "exec")
    loc = {}
    eval(comp, globals(), loc)
    return loc["f"](*args)


def fitness(m, c, cref, beta):
    """
    Compute the fitness of all elements of `c` and the length of `m` which is
    the number of multiplications
    """
    csubs = substitute(m, c)
    ftnss = np.zeros(len(c))
    for i in range(len(cref)):
        ftnss[i] = fitness_one_c(csubs[i], cref[i])
    score = np.sum(ftnss) / len(c) + beta / (len(m) + 1)
    num_mult = count_mult(c, len(m))
    return ftnss, num_mult, score


def fitness_one_c(csubs, cref):
    """
    How many operators are left when computing the difference `csubs - cref`.
    """
    eq = simplify(f"{csubs} - ({cref})")
    return 1 / (eq.count_ops() + 1)


def gen_c(m, num_vars):
    c = list()
    for i in range(num_vars):
        c.append([f"_c{i}", gen_one_c(m)])
    return c


def gen_func(m, c):
    dim = len(c)
    func = "def f("
    for i in range(dim):
        func += f"a{i}, "
    for i in range(dim):
        func += f"b{i}"
        if i + 1 < dim:
            func += ", "
    func += "):\n"
    for expr in m:
        func += f"    {expr[0]} = {' '.join(expr[1])}\n"
    for i in range(dim):
        func += f"    {c[i][0]} = {' '.join(c[i][1])}\n"
    func += "    return "
    for i in range(dim):
        func += f"_c{i}"
        if i + 1 < dim:
            func += ", "
    func += "\n"
    return func


def gen_individual(num_mult, dim):
    m = gen_m(num_mult, dim)
    c = gen_c(m, dim)
    return {"m": m, "c": c}


def gen_m(num_mult, num_vars):
    vars_a = [f"a{i}" for i in range(num_vars)]
    vars_b = [f"b{i}" for i in range(num_vars)]

    m = list()
    for i in range(num_mult):
        kind = choose([1, 2, 3, 4])
        if kind == 1:
            left = choose(vars_a)
            right = choose(vars_b)
            expr = [left, "*", right]
        elif kind == 2:
            left1, left2 = choose(vars_a, 2)
            op = choose(ADD_SUB)
            right = choose(vars_b)
            expr = ["(", left1, op, left2, ")", "*", right]
        elif kind == 3:
            left = choose(vars_a)
            op = choose(ADD_SUB)
            right1, right2 = choose(vars_b, 2)
            expr = [left, "*", "(", right1, op, right2, ")"]
        elif kind == 4:
            left1, left2 = choose(vars_a, 2)
            opl = choose(ADD_SUB)
            right1, right2 = choose(vars_b, 2)
            opr = choose(ADD_SUB)
            expr = ["(", left1, opl, left2, ")", "*", "(", right1, opr, right2, ")"]
        m.append([f"m{i}", expr, kind])
    return m


def gen_mat_equations(dim):
    mata = [[f"a{i+j*dim}" for i in range(dim)] for j in range(dim)]
    matb = [[f"b{i+j*dim}" for i in range(dim)] for j in range(dim)]
    matc = list()
    for row in range(dim):
        matc.append(list())
        for col in range(dim):
            matc[row].append("")
            for i in range(dim):
                matc[row][col] += mata[row][i] + "*" + matb[i][col]
                if i + 1 < dim:
                    matc[row][col] += "+"
    return matc, [element for row in matc for element in row]


def gen_one_c(m):
    vars_m = [f"m{i}" for i in range(len(m))]
    num_operands = np.random.randint(1, len(m) + 1)
    operands = choose(vars_m, num_operands)
    c = [operands[0]]
    if num_operands > 1:
        for i in range(1, num_operands):
            op = choose(ADD_SUB)
            c.append(op)
            c.append(operands[i])
    return c


def gen_var(letter, dim):
    return choose([f"{letter}{i+1}" for i in range(dim)])


def mutate(population, percent_mutation, num_vars, num_mult):
    for individual in population:
        if np.random.rand() <= percent_mutation / 100:
            m_mut = mutate_m(individual["m"], num_vars)
            c_mut = mutate_c(individual["c"], num_mult)
            individual["m"] = m_mut
            individual["c"] = c_mut
    return population


def mutate_c(c, num_mult):
    """
    Either swap two equations `ci`, `cj` or in an equation like `m1 + m3 - m4`,
    * either change one of the variable's index or
    * switch between `+` and `-` or
    * add a new term
    """
    idx, j = choose(np.arange(len(c)), 2)  # choose two of c0, c1, c2, ...
    _, eq = c[idx]
    kind = choose([1, 2, 3, 4])
    if kind == 1:  # switch `ci` and `cj`
        vari, eqi = c[idx]
        varj, eqj = c[j]
        c[idx] = [vari, eqj]
        c[j] = [varj, eqi]
        return c
    elif kind == 2:  # change a variable
        # if every variable is already used, we change an operator
        if len(eq) == 2 * num_mult - 1:
            num_ops = len(eq) // 2
            op_idx = choose(np.arange(num_ops)) * 2 + 1
            eq[op_idx] = change_op(eq[op_idx])
            c[idx][1] = eq
            return c
        num_vars = len(eq) // 2 + 1
        var_idx = choose(np.arange(num_vars)) * 2
        allowed_indices = np.ones(num_mult, dtype=bool)
        for i in range(num_vars):
            allowed_indices[int(eq[2 * i][1:])] = False
        m_new = f"m{choose(np.arange(num_mult)[allowed_indices])}"
        eq[var_idx] = m_new
        c[idx][1] = eq
        return c
    elif kind == 3:  # change an operator
        if len(eq) == 1:  # there is no operator
            # if there is no operator, we change the one variable
            c[idx][1][0] = change_var(eq[0], num_mult)
            return c
        num_ops = len(eq) // 2
        op_idx = choose(np.arange(num_ops)) * 2 + 1
        eq[op_idx] = change_op(eq[op_idx])
        c[idx][1] = eq
        return c
    elif kind == 4:  # add a variable
        if len(eq) == 2 * num_mult - 1:  # all `mi` are already present
            # see if maybe `c[j]` does not use all `mi`
            idx = j
            _, eq = c[idx]
            if len(eq) == 2 * num_mult - 1:  # `c[j]` also uses all `mi`
                # then at least change an operator
                num_ops = num_mult - 1
                op_idx = choose(np.arange(num_ops)) * 2 + 1
                eq[op_idx] = change_op(eq[op_idx])
                c[idx][1] = eq
                return c
        # at this point, we're either changing `c[idx]` or `c[j]` via `eq`
        num_vars = len(eq) // 2 + 1
        allowed_indices = np.ones(num_mult, dtype=bool)
        for i in range(num_vars):
            allowed_indices[int(eq[2 * i][1:])] = False
        m_new = f"m{choose(np.arange(num_mult)[allowed_indices])}"
        eq.extend([choose(ADD_SUB), m_new])  # add a new variable
        c[idx][1] = eq
        assert len(eq) <= 2 * num_mult - 1
        return c


def mutate_m(m, dim):
    """
    In an equation like `a2 * (b1 + b3)`, either change one of the variable's
    index or switch between `+` and `-`.
    """
    idx = choose(np.arange(len(m)))  # choose one of m0, m1, m2, ...
    _, expr, kind = m[idx]
    if kind == 1:
        if choose([False, True]):  # left
            expr[0] = change_var(expr[0], dim)
        else:  # right
            expr[2] = change_var(expr[2], dim)
    elif kind == 2:
        what = choose([1, 2, 3, 4])
        if what == 1:  # a_left
            expr[1] = change_var(expr[1], dim)
        elif what == 2:  # operator
            expr[2] = change_op(expr[2])
        elif what == 3:  # a_right
            expr[3] = change_var(expr[3], dim)
        elif what == 4:  # b
            expr[6] = change_var(expr[6], dim)
    elif kind == 3:
        what = choose([1, 2, 3, 4])
        if what == 1:  # a
            expr[0] = change_var(expr[0], dim)
        elif what == 2:  # b_left
            expr[3] = change_var(expr[3], dim)
        elif what == 3:  # operator
            expr[4] = change_op(expr[4])
        elif what == 4:  # b_right
            expr[5] = change_var(expr[5], dim)
    elif kind == 4:
        what = choose([1, 2, 3, 4, 5, 6])
        if what == 1:  # a_left
            expr[1] = change_var(expr[1], dim)
        elif what == 2:  # operator
            expr[2] = change_op(expr[2])
        elif what == 3:  # a_right
            expr[3] = change_var(expr[3], dim)
        elif what == 4:  # b_left
            expr[7] = change_var(expr[7], dim)
        elif what == 5:  # operator
            expr[8] = change_op(expr[8])
        elif what == 6:  # b_right
            expr[9] = change_var(expr[9], dim)
    return m


def next_gen(population, percent_elite, percent_mutation, num_vars, num_mult):
    population_size = len(population)

    nr_elite = int(np.ceil(percent_elite * population_size / 100))

    # the elite survive unchanged
    population_new = deepcopy(population[-nr_elite:])

    # fittest are most likely to be selected
    selection_probabilities = (np.arange(population_size) + 1) / (
        population_size * (population_size + 1) / 2
    )

    offspring = breed(
        population,
        selection_probabilities,
        n_offspring=population_size - nr_elite,
    )

    offspring = mutate(offspring, percent_mutation, num_vars, num_mult)

    population_new.extend(offspring)

    return population_new


def work(individual, cref, beta):
    return fitness(individual["m"], individual["c"], cref, beta)


def population_fitness(population, cref, beta):
    population_size = len(population)

    ftnss = np.zeros((population_size, len(population[0]["c"])))
    num_mults = np.zeros(population_size, dtype=int)
    scores = np.zeros(population_size)

    for i, individual in enumerate(population):
        ftnss_i, num_mult_i, score_i = fitness(
            individual["m"], individual["c"], cref, beta
        )
        ftnss[i, :] = ftnss_i
        num_mults[i] = num_mult_i
        scores[i] = score_i

    # worker = partial(work, cref=cref, beta=beta)
    # with ProcessPoolExecutor() as ex:
    #     results = list(ex.map(worker, population))
    # for i, result in enumerate(results):
    #     ftnss[i, :] = result[0]
    #     num_mults[i] = result[1]
    #     scores[i] = result[2]
    return ftnss, np.sum(ftnss, axis=1) / len(cref), num_mults, scores


def population_init(population_size, num_mult, dim):
    return [gen_individual(num_mult, dim) for _ in range(population_size)]


def sort_by(by, population, num_mults):
    idx_sorted = np.argsort(by)
    return (
        by[idx_sorted],
        [population[idx] for idx in idx_sorted],
        num_mults[idx_sorted],
        idx_sorted,
    )


def stats(scores, num_mults, sum_fitness, pm):
    """
    Compute some statistics.
    """
    scores = np.array(scores)

    pm["score min"].append(np.min(scores))
    pm["score 10%"].append(np.quantile(scores, 0.1))
    pm["score mean"].append(np.mean(scores))
    pm["score 90%"].append(np.quantile(scores, 0.9))
    pm["score max"].append(np.max(scores))

    # the best individuals' sum_fitness
    pm["sum_fitness"].append([f"{x:.2f}" for x in sum_fitness[-3:]])

    # the best individuals' numbers of Mult nodes
    pm["num_mult"].append(num_mults[-3:])

    return pm


def substitute(m, c):
    """
    Plug all m-equations into c-equations.
    """
    csubs = list()
    for _, m_ in c:
        # e.g., m_ = ['m0', '-', 'm2']
        s = ""
        for k, mi in enumerate(m_):
            if k % 2 == 0:
                i = int(mi[1:])
                s += "(" + "".join(m[i][1]) + ")"
            else:
                s += mi
        csubs.append(s)
    return csubs


###########################################################################


def test_fitness_one_c():
    print("  test_fitness_one_c()", end="", flush=True)
    csubs = "a0*b0+a2*b1"
    cref = "a0*b0+a2*b1"
    f_got = fitness_one_c(csubs, cref)
    f_exp = 1 / (0 + 1)
    assert f_got == f_exp, f"Got fitness {f_got} but expected {f_exp}."
    csubs = "a0*b0+a2*b1"
    cref = "a0*b0-a2*b1"
    f_got = fitness_one_c(csubs, cref)
    f_exp = 1 / (2 + 1)
    assert f_got == f_exp, f"Got fitness {f_got} but expected {f_exp}."
    csubs = "a0*b0-a2*b1"
    cref = "a0*b0+a2*b1"
    f_got = fitness_one_c(csubs, cref)
    f_exp = 1 / (3 + 1)
    assert f_got == f_exp, f"Got fitness {f_got} but expected {f_exp}."
    print("\r✓ test_fitness_one_c()")


def test_gen_mat_equations():
    print("  test_gen_mat_equations()", end="", flush=True)
    dim = 2
    matc = gen_mat_equations(dim)[0]
    # [a0 , a1] [b0 , b1] = [a0 * b0 + a1 * b2 , a0 * b1 + a1 * b3]
    # [a2 , a3] [b2 , b3]   [a2 * b0 + a3 * b2 , a2 * b1 + a3 * b3]
    matc_expected = [["a0*b0+a1*b2", "a0*b1+a1*b3"], ["a2*b0+a3*b2", "a2*b1+a3*b3"]]
    for row in range(dim):
        for col in range(dim):
            assert matc[row][col] == matc_expected[row][col], f"[{row}, {col}]"

    dim = 3
    matc = gen_mat_equations(dim)[0]
    # [a0 , a1 , a2] [b0 , b1 , b2] =
    #   [a0 * b0 + a1 * b3 + a2 * b6 , a0 * b1 + a1 * b4 + a2 * b7 , a0 * b2 + a1 * b5 + a2 * b8]
    # [a3 , a4 , a5] [b3 , b4 , b5]
    #   [a3 * b0 + a4 * b3 + a5 * b6 , a3 * b1 + a4 * b4 + a5 * b7 , a3 * b2 + a4 * b5 + a5 * b8]
    # [a6 , a7 , a8] [b6 , b7 , b8]
    #   [a6 * b0 + a7 * b3 + a8 * b6 , a6 * b1 + a7 * b4 + a8 * b7 , a6 * b2 + a7 * b5 + a8 * b8]
    matc_expected = [
        ["a0*b0+a1*b3+a2*b6", "a0*b1+a1*b4+a2*b7", "a0*b2+a1*b5+a2*b8"],
        ["a3*b0+a4*b3+a5*b6", "a3*b1+a4*b4+a5*b7", "a3*b2+a4*b5+a5*b8"],
        ["a6*b0+a7*b3+a8*b6", "a6*b1+a7*b4+a8*b7", "a6*b2+a7*b5+a8*b8"],
    ]
    for row in range(dim):
        for col in range(dim):
            assert matc[row][col] == matc_expected[row][col], f"[{row}, {col}]"
    print("\r✓ test_gen_mat_equations()")


def test_simplify():
    print("  test_simplify()", end="", flush=True)
    matc = gen_mat_equations(2)[1]
    strassen = [
        "(a0 + a3) * (b0 + b3) + a3 * (b2 - b0) - (a0 + a1) * b3 + (a1 - a3) * (b2 + b3)",
        "a0 * (b1 - b3) + (a0 + a1) * b3",
        "(a2 + a3) * b0 + a3 * (b2 - b0)",
        "(a0 + a3) * (b0 + b3) - (a2 + a3) * b0 + a0 * (b1 - b3) + (a2 - a0) * (b0 + b1)",
    ]
    for i in range(4):
        eq = simplify(matc[i] + "-(" + strassen[i] + ")")
        assert eq.count_ops() == 0
    print("\r✓ test_simplify()")


def test_substitute():
    print("  test_substitute()", end="", flush=True)
    m = [
        ["m0", ["(", "a1", "-", "a2", ")", "*", "(", "b1", "-", "b2", ")"], 4],
        ["m1", ["a1", "*", "(", "b1", "-", "b0", ")"], 3],
    ]
    c = [["_c0", ["m0", "+", "m1"]], ["_c1", ["m0"]]]
    csubs = substitute(m, c)
    csubs_expected = ["((a1-a2)*(b1-b2))+(a1*(b1-b0))", "((a1-a2)*(b1-b2))"]
    for i in range(len(c)):
        assert (
            csubs[i] == csubs_expected[i]
        ), f"Got {csubs[i]} | Expected {csubs_expected[i]}"
    print("\r✓ test_substitute()")


def playground():
    dim = 3
    m_dad = gen_m(dim**2, dim)
    m_mom = gen_m(dim**2, dim)
    m_child = cross_m(m_dad, m_mom)
    print("m_dad:")
    for m in m_dad:
        print("  ", m)
    print("m_mom:")
    for m in m_mom:
        print("  ", m)
    print("m_child:")
    for m in m_child:
        print("  ", m)
    for i in range(dim, 10):
        pass
        # m = gen_m(i, dim)
        # print(m)
        # c = gen_c(m)
        # print(c)

        # m = gen_m(i, dim)
        # c = gen_c(m, dim)
        # csubs = substitute(m, c)
        # for cs in csubs:
        #     print(cs)

        # print("-" * 10)
        # m = gen_m(i, dim)
        # c = gen_c(m, dim)
        # func = gen_func(m, c)
        # print(func)
        # print(evaluate(func, 1, 2, 3, 4, 5, 6))
        # m = mutate_m(m, dim)
        # func = gen_func(m, c)
        # print(func)
        # print(evaluate(func, 1, 2, 3, 4, 5, 6))

        # print("-" * 10)
        # m = gen_m(i, dim)
        # c = gen_c(m, dim)
        # func = gen_func(m, c)
        # print(func)
        # print(evaluate(func, 1, 2, 3, 4, 5, 6))
        # c = mutate_c(c, len(m))
        # func = gen_func(m, c)
        # print(func)
        # print(evaluate(func, 1, 2, 3, 4, 5, 6))

        # print("-" * 10)
        # m_dad = gen_m(i, dim)
        # c_dad = gen_c(m_dad, dim)
        # print("c_dad", c_dad)
        # m_mom = gen_m(i, dim)
        # c_mom = gen_c(m_mom, dim)
        # print("c_mom", c_mom)
        # c_child = cross_c(c_mom, c_dad)
        # print("c_child", c_child)

        # print("-" * 10)
        # i_dad, i_mom = choose(np.arange(2, 5), 2)
        # m_dad = gen_m(i_dad, dim)
        # print("m_dad", m_dad)
        # m_mom = gen_m(i_mom, dim)
        # print("m_mom", m_mom)
        # m_child = cross_m(m_dad, m_mom)
        # print("m_child", m_child)

        # print("-" * 10)
        # m = gen_m(i, dim)
        # c = gen_c(m, dim)
        # csubs = substitute(m, c)
        # mat_eq = gen_mat_equations(dim)[1]
        # for i, cs in enumerate(csubs):
        #     print(fitness_one_c(cs, mat_eq[i]))


if __name__ == "__main__":
    test_fitness_one_c()
    test_gen_mat_equations()
    test_simplify()
    test_substitute()
    playground()
