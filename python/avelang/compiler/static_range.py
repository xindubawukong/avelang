"""Expand literal or captured-constexpr loops before resolving immediates."""

import ast
import copy


class _ReplaceIndex(ast.NodeTransformer):
    def __init__(self, name, value):
        self.name, self.value = name, value

    def visit_Name(self, node):
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(self.value), node)
        return node


class _Expand(ast.NodeTransformer):
    def __init__(self, constants):
        self.constants = {k: v.get("value") if isinstance(v, dict) else v for k, v in constants.items()}

    def bound(self, node):
        if isinstance(node, ast.Name) and node.id in self.constants:
            return self.constants[node.id]
        return ast.literal_eval(node)

    def visit_For(self, node):
        call = node.iter
        if not (
            isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "static_range"
        ):
            return self.generic_visit(node)
        if not isinstance(node.target, ast.Name) or node.orelse or call.keywords:
            raise ValueError("static_range requires a scalar loop index and no for-else")
        try:
            bounds = [self.bound(arg) for arg in call.args]
            if not all(type(value) is int for value in bounds):
                raise ValueError()
            values = range(*bounds)
        except (TypeError, ValueError) as error:
            raise ValueError("static_range bounds must be integer literals or constexpr integers") from error
        if len(values) > 256:
            raise ValueError("static_range is limited to 256 iterations")
        for stmt in node.body:
            for child in ast.walk(stmt):
                if isinstance(child, (ast.Break, ast.Continue)):
                    raise ValueError("static_range does not support break or continue")  # noqa: TRY004
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id == node.target.id:
                    raise ValueError("static_range index cannot be reassigned or shadowed")
        result = []
        for value in values:
            # Preserve the loop variable's value after a nonempty loop.
            assignment = ast.Assign(targets=[copy.deepcopy(node.target)], value=ast.Constant(value))
            result.append(ast.copy_location(assignment, node))
            body = []
            for stmt in node.body:
                expanded = _ReplaceIndex(node.target.id, value).visit(copy.deepcopy(stmt))
                expanded = self.visit(expanded)
                body.extend(expanded if isinstance(expanded, list) else [expanded])
            # Ave gives loop-local tensor views their own symbol scope.
            # Preserve that scope after expansion so a view can be rebound
            # on the next iteration instead of becoming a tensor copy.
            scope = ast.If(test=ast.Constant(True), body=body or [ast.Pass()], orelse=[])
            result.append(ast.copy_location(scope, node))
        return result


def expand_static_ranges(function, constants=None):
    return ast.fix_missing_locations(_Expand(constants or {}).visit(copy.deepcopy(function)))
