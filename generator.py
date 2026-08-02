from sympy import simplify
import numpy as np

ADD_SUB = ["+", "-"]


def choose(*args, **kwargs):
    return np.random.choice(*args, **kwargs, replace=False)


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
    The child has a number of multiplications between dad's and mom's numbers.
    The equations randomly come from either parent.
    """
    m_concat = [*m_dad, *m_mom]
    np.random.shuffle(m_concat)

    num_mult_dad = len(m_dad)
    num_mult_mom = len(m_mom)
    num_mult_min = min(num_mult_dad, num_mult_mom)
    num_mult_max = max(num_mult_dad, num_mult_mom)
    num_mult_child = choose(np.arange(num_mult_min, num_mult_max + 1))

    m_child = list()
    for i in range(num_mult_child):
        _, eq, kind = m_concat[i]
        m_child.append([f"m{i}", eq, kind])
    return m_child


def evaluate(func, *args):
    comp = compile(func, "", "exec")
    loc = {}
    eval(comp, globals(), loc)
    return loc["f"](*args)


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


def gen_c(m, dim):
    c = list()
    for i in range(dim):
        c.append([f"_c{i}", gen_one_c(m)])
    return c


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


def gen_m(num_mult, dim):
    vars_a = [f"a{i}" for i in range(dim)]
    vars_b = [f"b{i}" for i in range(dim)]

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


def gen_var(letter, dim):
    return choose([f"{letter}{i+1}" for i in range(dim)])


def change_op(op):
    if op == "+":
        return "-"
    else:
        return "+"


def change_var(var, dim):
    vars = [f"{var[0]}{i}" for i in range(dim)]
    opt1, opt2 = choose(vars, 2)
    print(opt1, opt2, var)
    if opt1 == var:
        return opt2
    else:
        return opt1


def mutate_c(c, num_mult):
    """
    In an equation like `m1 + m3 - m4`, either change one of the variable's
    index or switch between `+` and `-`.
    """
    idx, j = choose(np.arange(len(c)), 2)  # choose two of c0, c1, c2, ...
    _, eq = c[idx]
    kind = choose([1, 2, 3])
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


def substitute(m, c):
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


###########################################################################
def main():
    dim = 3
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

        print("-" * 10)
        i_dad, i_mom = choose(np.arange(2, 5), 2)
        m_dad = gen_m(i_dad, dim)
        print("m_dad", m_dad)
        m_mom = gen_m(i_mom, dim)
        print("m_mom", m_mom)
        m_child = cross_m(m_dad, m_mom)
        print("m_child", m_child)


if __name__ == "__main__":
    test_gen_mat_equations()
    test_simplify()
    test_substitute()
    print(gen_mat_equations(2))
    main()
