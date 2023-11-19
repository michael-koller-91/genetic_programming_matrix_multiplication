import ast
import copy
import inspect
import textwrap
import numpy as np


class GetSubtree(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.node_nr = 0
        self.node_counter = 0
        self.node = None

    def __call__(self, tree, node_nr):
        self.node_nr = node_nr
        self.node_counter = 0
        self.visit(tree)
        return self.node

    def visit_BinOp(self, node):
        self.node_counter += 1
        if self.node_counter != self.node_nr:
            self.generic_visit(node)
            return node
        else:
            if type(node.op) in [ast.Add, ast.Sub, ast.Mult]:
                self.node = node
            else:
                raise ValueError("node.op not in [Add, Sub, Mult]: node.op =", node.op)
            return node


class ReplaceSubtree(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.node_nr = 0
        self.node_counter = 0
        self.binop = [ast.Add, ast.Sub, ast.Mult]
        self.node = None
        self.left = None

    def __call__(self, tree, node_nr, node):
        self.node_nr = node_nr
        self.node_counter = 0
        self.node = node
        self.visit(tree)
        return tree

    def visit_BinOp(self, node):
        self.node_counter += 1
        if self.node_counter != self.node_nr:
            self.generic_visit(node)
            return node
        else:
            if type(node.op) in self.binop:
                node = self.node
            return node


get_subtree = GetSubtree()
replace_subtree = ReplaceSubtree()


def count_nodes(tree, node_type):
    """
    Count the number of nodes of type `node_type` in the AST `tree`.
    """
    counter = 0
    for node in ast.walk(tree):
        if type(node) is node_type:
            counter += 1
    return counter


def crossover(parent_1, parent_2, verbose=False, rng=np.random.default_rng()):
    """
    Select from `parent_1` a random subtree `st_1` and select from `parent_2` a
    random subtree `st_2`. Replace `st_1` with `st_2`. Return the new
    `parent_1` as the offspring.
    """
    # do not modify parent_1
    offspring = copy.deepcopy(parent_1)

    n_Assign_1 = len(offspring.body[0].body) - 1
    n_Assign_2 = len(parent_2.body[0].body) - 1
    assert n_Assign_1 == n_Assign_2

    # choose a random c_i
    i_Assign_1 = rng.choice(n_Assign_1)
    i_Assign_2 = rng.choice(n_Assign_2)
    if verbose:
        print(f"from parent_1 select element c{i_Assign_1+1}")
        print(f"from parent_2 select element c{i_Assign_2+1}")
    Assign_1 = offspring.body[0].body[i_Assign_1]
    Assign_2 = parent_2.body[0].body[i_Assign_2]

    n_BinOp_1 = count_nodes(Assign_1, ast.BinOp)
    n_BinOp_2 = count_nodes(Assign_2, ast.BinOp)

    n_st_1 = rng.choice(n_BinOp_1) + 1
    n_st_2 = rng.choice(n_BinOp_2) + 1
    if verbose:
        print("from parent_1 select node", n_st_1)

    # get parent_2's subtree
    st_2 = copy.deepcopy(get_subtree(Assign_2, n_st_2))
    if verbose:
        print("from parent_2 select subtree", tree_to_source(st_2))

    # replace parent_1's subtree with parent_2's subtree
    new_Assign_1 = replace_subtree(Assign_1, n_st_1, st_2)

    # put switched the new Assign into parent_1 trees
    offspring.body[0].body[i_Assign_1] = new_Assign_1

    return offspring


def mutate(tree_in, copy_tree=False, verbose=False, rng=np.random.default_rng()):
    """
    Mutate a random BinOp of the AST `tree_in`.
    """

    if copy_tree:
        tree = copy.deepcopy(tree_in)
    else:
        tree = tree_in

    n_BinOp = count_nodes(tree, ast.BinOp)
    n_mutate = rng.choice(n_BinOp)

    counter = 0
    for node in ast.walk(tree):
        if type(node) is ast.BinOp:
            if counter == n_mutate:
                if type(node.op) is ast.Add:
                    node.op = rng.choice([ast.Sub, ast.Mult])()
                    verbose and print(
                        f"{mutate.__name__}(): {ast.Add} -> {type(node.op)}"
                    )
                elif type(node.op) is ast.Sub:
                    node.op = rng.choice([ast.Add, ast.Mult])()
                    verbose and print(
                        f"{mutate.__name__}(): {ast.Sub} -> {type(node.op)}"
                    )
                elif type(node.op) is ast.Mult:
                    node.op = rng.choice([ast.Add, ast.Sub])()
                    verbose and print(
                        f"{mutate.__name__}(): {ast.Mult} -> {type(node.op)}"
                    )

                else:
                    raise ValueError(
                        "node.op not in [Add, Sub, Mult]: node.op =", node.op
                    )
            else:
                counter += 1

    return tree


def func_to_source(func):
    """
    Return the source code of the function `func`.
    """
    return textwrap.dedent(inspect.getsource(func))


def func_to_tree(func):
    """
    Return the AST of the function `func`.
    """
    return ast.parse(func_to_source(func))


def tree_to_source(tree):
    """
    Return the source code of the AST `tree`.
    """
    return ast.unparse(ast.fix_missing_locations(tree))


#
# functions for tests
#
def f_test_crossover_1(a, b):
    c1 = a - b
    c2 = a + b
    return c1, c2


def f_test_crossover_2(a, b):
    c1 = (a + b * a) * b + a * (b + a)
    c2 = (a * b + a + b) * a + b * a
    return c1, c2


# this function is expected with the given fixed random seed
def f_test_crossover_expected(a, b):
    c1 = a - b
    c2 = b * a
    return (c1, c2)


def f_test_mutate(a, b):
    c = a + b
    return c


# this function is expected with the given fixed random seed
def f_test_mutate_expected(a, b):
    c = a * b
    return c


if __name__ == "__main__":
    rng = np.random.default_rng(123456789)

    #
    # test crossover()
    #
    parent_1 = func_to_tree(f_test_crossover_1)
    parent_2 = func_to_tree(f_test_crossover_2)
    offspring = crossover(parent_1, parent_2, rng=rng)
    tree_source = tree_to_source(offspring)

    expected_source = func_to_source(f_test_crossover_expected)
    code_1 = textwrap.dedent(tree_source.split(":")[1].strip("\n"))
    code_2 = textwrap.dedent(expected_source.split(":")[1].strip("\n"))
    if code_1 == code_2:
        print("crossover(): PASS")
    else:
        print("crossover(): FAIL")
        print("-" * 10, "before crossover():", "-" * 10)
        print(tree_to_source(parent_1))
        print(tree_to_source(parent_2))
        print("\n", "-" * 10, "after crossover():", "-" * 10)
        print(tree_source)

    #
    # test mutate()
    #
    tree = func_to_tree(f_test_mutate)
    tree_source_before = tree_to_source(tree)
    tree = mutate(tree, rng=rng)
    tree_source = tree_to_source(tree)

    expected_source = func_to_source(f_test_mutate_expected)
    code_1 = textwrap.dedent(tree_source.split(":")[1].strip("\n"))
    code_2 = textwrap.dedent(expected_source.split(":")[1].strip("\n"))
    if code_1 == code_2:
        print("mutate(): PASS")
    else:
        print("mutate(): FAIL")
        print("-" * 10, "before mutate():", "-" * 10)
        print(tree_source_before)
        print("\n", "-" * 10, "after mutate():", "-" * 10)
        print(tree_source)
