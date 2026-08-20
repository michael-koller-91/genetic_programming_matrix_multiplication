import ast
import inspect
import numpy as np
import numba
import textwrap
import utils as ut


def play1():
    expr = """
def f(a):
    print('inside f')
    return a + 1
"""

    def c(expr):
        comp = compile(expr, "", "exec")
        loc = {}
        eval(comp, globals(), loc)
        print("loc['f'](3):", loc["f"](3))

    c(expr)


def play2():
    def f(a, b):
        c = a + b
        return c

    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    print(ut.count_nodes(tree, ast.BinOp))


def play3():
    def f(a, b):
        c = a + b
        return c

    def append_source(filename, source):
        with open(filename, "a") as fl:
            fl.write(source)
            fl.write("\n\n")

    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))

    with open("play3.txt", "w") as fl:
        fl.write("\n")

    append_source("play3.txt", ut.tree_to_source(tree))
    append_source("play3.txt", ut.tree_to_source(tree))


def play4():
    @numba.njit
    def strassen_2x2_v1(a1, a2, a3, a4, b1, b2, b3, b4):
        m1 = (a1 + a4) * (b1 + b4)
        m2 = (a3 + a4) * b1
        m3 = a1 * (b2 - b4)
        m4 = a4 * (b3 - b1)
        m5 = (a1 + a2) * b4
        m6 = (a3 - a1) * (b1 + b2)
        m7 = (a2 - a4) * (b3 + b4)
        c1 = m1 + m4 - m5 + m7
        c2 = m3 + m5
        c3 = m2 + m4
        c4 = m1 - m2 + m3 + m6
        return c1, c2, c3, c4

    @numba.njit
    def strassen_2x2_v2(a1, a2, a3, a4, b1, b2, b3, b4):
        c1 = (
            (a1 + a4) * (b1 + b4)
            + a4 * (b3 - b1)
            - (a1 + a2) * b4
            + (a2 - a4) * (b3 + b4)
        )
        c2 = a1 * (b2 - b4) + (a1 + a2) * b4
        c3 = (a3 + a4) * b1 + a4 * (b3 - b1)
        c4 = (
            (a1 + a4) * (b1 + b4)
            - (a3 + a4) * b1
            + a1 * (b2 - b4)
            + (a3 - a1) * (b1 + b2)
        )
        return c1, c2, c3, c4

    sig = tuple(list(numba.types.int64 for _ in range(8)))
    strassen_2x2_v1.compile(sig)
    strassen_2x2_v2.compile(sig)
    v1 = strassen_2x2_v1.inspect_llvm(sig)
    print(v1.count("mul"))
    v2 = strassen_2x2_v2.inspect_llvm(sig)
    print(v2.count("mul"))


def play5():
    expr = """
@numba.njit
def f(a):
    return a + 1
"""
    ns = {"numba": numba}
    exec(expr, ns, ns)
    print(ns["f"].compile((numba.types.int64,)))


def play6():
    from sympy import simplify

    s1 = (
        "(a1 + a4) * (b1 + b4) + a4 * (b3 - b1) "
        " - (a1 + a2) * b4 + (a2 - a4) * (b3 + b4)"
        "- a1 * b1 - a2 * b3"
    )
    s2 = (
        "(a1 - a4) * (b1 + b4) + a4 * (b3 - b1) "
        " - (a1 + a2) * b4 + (a2 - a4) * (b3 + b4)"
        "- a1 * b1 - a2 * b3"
    )
    ss1 = simplify(s1)
    print("result:", ss1)
    print("num ops:", ss1.count_ops())
    ss2 = simplify(s2)
    print("result:", ss2)
    print("num ops:", ss2.count_ops())


if __name__ == "__main__":
    # play1()
    # play2()
    # play3()
    # play4()
    # play5()
    play6()
