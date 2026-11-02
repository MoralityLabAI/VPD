"""Bounded literal/arithmetic evaluator for upstream verifier eval call sites.

No Python bytecode, names, calls, attributes, or comprehensions are executed.
This is an explicitly reported verifier adapter; upstream source stays pinned.
"""
import ast
import math
import operator

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
       ast.Mod: operator.mod, ast.Pow: operator.pow}
COMPARISONS = {ast.Eq: operator.eq, ast.NotEq: operator.ne,
               ast.Lt: operator.lt, ast.LtE: operator.le,
               ast.Gt: operator.gt, ast.GtE: operator.ge}


def safe_eval(expression):
    if not isinstance(expression, str) or len(expression) > 50_000:
        raise ValueError('Verifier expression exceeds text bound')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 12_000:
        raise ValueError('Verifier expression exceeds node bound')

    def bounded(value):
        if type(value) is int and value.bit_length() > 4096:
            raise ValueError('Verifier integer exceeds bound')
        if type(value) is float and not math.isfinite(value):
            raise ValueError('Nonfinite verifier number')
        return value

    def visit(node, depth=0):
        if depth > 80:
            raise ValueError('Verifier expression exceeds depth bound')
        child = lambda n: visit(n, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
            return bounded(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            factory = {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)]
            return factory(child(n) for n in node.elts)
        if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
            return {child(k): child(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = child(node.operand)
            if type(value) not in (int, float):
                raise ValueError('Numeric operands required')
            return bounded(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = child(node.left), child(node.right)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ValueError('Numeric operands required')
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError('Verifier exponent exceeds bound')
            value = OPS[type(node.op)](left, right)
            if type(value) not in (int, float):
                raise ValueError('Real numeric result required')
            return bounded(value)
        if isinstance(node, ast.Compare) and all(type(op) in COMPARISONS for op in node.ops):
            left = child(node.left)
            for op, term in zip(node.ops, node.comparators):
                right = child(term)
                if not COMPARISONS[type(op)](left, right):
                    return False
                left = right
            return True
        raise ValueError('Unsupported verifier expression: ' + type(node).__name__)

    return visit(tree.body)
