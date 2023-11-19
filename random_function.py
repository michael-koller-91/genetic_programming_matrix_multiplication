import ast
import numpy as np
import utils as ut


class RandomFunctionTree(ast.NodeTransformer):
    def __init__(self, dimensions=2) -> None:
        super().__init__()

        self.dimensions = dimensions
        # id: the variables a1, a2, ..., b1, b2, ...
        self.id = [f"a{i}" for i in range(1, self.dimensions**2 + 1)]
        self.id.extend([f"b{i}" for i in range(1, self.dimensions**2 + 1)])

        # define an empty function:
        # def f(a1, a2, ..., b1, b2, ...):
        #    return c1, c2, ...
        expr = "def f(" + self.id[0]
        for id in self.id[1:]:
            expr += ", " + id
        expr += "):"
        expr += "\n\treturn "
        for i in range(1, self.dimensions**2 + 1):
            expr += f"c{i}, "
        self.expr = expr[:-2]

        self.init_tree_body()

    def init_tree_body(self):
        self.tree = ast.parse(self.expr)
        self.body = self.tree.body[0].body

    def __call__(self, max_nodes_per_id=2):
        if max_nodes_per_id < 2:
            raise ValueError("Expected max_nodes_per_id to be greater than 1.")

        self.init_tree_body()
        for i in range(self.dimensions**2, 0, -1):
            self.assign(f"c{i}", max_nodes_per_id)
        return self.tree

    def __repr__(self):
        return ast.dump(self.tree, indent=4)

    def random_binop(self):
        return np.random.choice([ast.Add, ast.Sub, ast.Mult])()

    def random_id(self):
        return np.random.choice(self.id)

    def split_number(self, number):
        """
        Of the `number` nodes, create a random number of nodes in the left
        branch and the rest in the right branch.
        """
        number_left = np.random.randint(low=1, high=number)
        return number_left, number - number_left

    def assign(self, id, max_nodes_per_id):
        nr_nodes = np.random.randint(low=2, high=max_nodes_per_id + 1)
        value = self.append_node(nr_nodes)
        assign = ast.Assign(targets=[ast.Name(id=id, ctx=ast.Store())], value=value)
        self.body.insert(0, assign)

    def append_node(self, number):
        if number > 1:
            nr1, nr2 = self.split_number(number)
            node_left = self.append_node(nr1)
            node_right = self.append_node(nr2)
            node_op = self.random_binop()
            return ast.BinOp(left=node_left, op=node_op, right=node_right)
        elif number == 1:
            return ast.Name(id=self.random_id(), ctx=ast.Load())
        else:
            raise ValueError(
                f"Expected number to be larger than 1 but number = {number}."
            )


if __name__ == "__main__":
    random_function_tree = RandomFunctionTree()

    for i in range(3):
        max_nodes_per_id = 3 + i
        print("-" * 10, f"max_nodes_per_id = {max_nodes_per_id}", "-" * 10)
        print(ut.tree_to_source(random_function_tree(max_nodes_per_id)))
