import numpy as np

ADD_SUB = ["+", "-"]


def choose(*args, **kwargs):
    return np.random.choice(*args, **kwargs, replace=False)


def gen_c(m, dim):
    vars_m = [f"m{i}" for i in range(len(m))]
    num_operands = np.random.randint(1, len(m) + 1)
    operands = np.sort(choose(vars_m, num_operands))
    c = [operands[0]]
    if num_operands > 1:
        for i in range(1, num_operands):
            op = choose(ADD_SUB)
            c.append(op)
            c.append(operands[i])
    return c


def gen_func(num_mult, dim):
    m = gen_m(num_mult, dim)
    func = "def func("
    for i in range(dim):
        func += f"a{i}, "
    for i in range(dim):
        func += f"b{i}"
        if i + 1 < dim:
            func += ", "
    func += "):\n"
    for expr in m:
        func += f"    {expr[0]} = {' '.join(expr[1])}\n"
    c = list()
    for i in range(dim):
        c.append([f"_c{i}", gen_c(m, dim)])
        func += f"    {c[i][0]} = {' '.join(c[i][1])}\n"
    func += "    return "
    for i in range(dim):
        func += f"_c{i}"
        if i + 1 < dim:
            func += ", "
    func += "\n"
    return func, m, c


def gen_m(num_mult, dim):
    vars_a = [f"a{i}" for i in range(dim)]
    vars_b = [f"b{i}" for i in range(dim)]

    m = list()
    for i in range(num_mult):
        c = np.random.choice([1, 2, 3, 4])
        if c == 1:
            left = choose(vars_a)
            right = choose(vars_b)
            expr = [left, "*", right]
        elif c == 2:
            left1, left2 = choose(vars_a, 2)
            op = choose(ADD_SUB)
            right = choose(vars_b)
            expr = ["(", left1, op, left2, ")", "*", right]
        elif c == 3:
            left = choose(vars_a)
            op = choose(ADD_SUB)
            right1, right2 = choose(vars_b, 2)
            expr = [left, "*", "(", right1, op, right2, ")"]
        elif c == 4:
            left1, left2 = choose(vars_a, 2)
            opl = choose(ADD_SUB)
            right1, right2 = choose(vars_b, 2)
            opr = choose(ADD_SUB)
            expr = ["(", left1, opl, left2, ")", "*", "(", right1, opr, right2, ")"]
        m.append([f"m{i}", expr, c])
    return m


def gen_var(letter, dim):
    return np.random.choice([f"{letter}{i+1}" for i in range(dim)])


def main():
    dim = 3
    for i in range(dim, 10):
        # m = gen_m(i, dim)
        # print(m)
        # c = gen_c(m)
        # print(c)
        func, m, c = gen_func(dim, dim)
        print(func)
        print(m)
        print(c)


if __name__ == "__main__":
    main()
