"""Deterministic, pre-validation repairs for generated candidate source."""
from __future__ import annotations

import ast

from .safety import ALLOWED_IMPORTS, TEST_ONLY_IMPORTS, is_allowed_import


class _Transformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.helpers: dict[tuple[str, str], str] = {}
        self.evaluation = False

    def visit_Import(self, node):  # noqa: N802
        node.names = [a for a in node.names if is_allowed_import(a.name, ALLOWED_IMPORTS)]
        return node if node.names else None

    def visit_ImportFrom(self, node):  # noqa: N802
        return node if not node.level and is_allowed_import(node.module or "", ALLOWED_IMPORTS) else None

    def visit_Call(self, node):  # noqa: N802
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "float" and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Attribute)
                and isinstance(node.args[0].func.value, ast.Name)
                and node.args[0].func.value.id == "context"
                and node.args[0].func.attr == "evaluate_validation"):
            self.evaluation = True
            return ast.copy_location(ast.Call(ast.Name("_autofix_primary", ast.Load()), node.args, []), node)

        # Guard against FMRanker(dimension=train_x.shape[1])
        is_fm_ranker = (
            (isinstance(node.func, ast.Name) and node.func.id == "FMRanker")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "FMRanker")
        )
        if is_fm_ranker:
            if node.args and _is_shape_one_expr(node.args[0]):
                node.args[0] = ast.copy_location(ast.Name("_field_dim", ast.Load()), node.args[0])
            for kw in node.keywords:
                if kw.arg == "dimension" and _is_shape_one_expr(kw.value):
                    kw.value = ast.copy_location(ast.Name("_field_dim", ast.Load()), kw.value)

        if not isinstance(node.func, ast.Name) or node.func.id not in {"getattr", "hasattr"}:
            return node
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
            return node
        attr = node.args[1].value
        if not isinstance(attr, str) or not attr.isidentifier():
            return node
        kind = node.func.id
        if kind == "getattr" and len(node.args) == 2:
            return ast.copy_location(ast.Attribute(node.args[0], attr, ast.Load()), node)
        if (kind == "getattr" and len(node.args) != 3) or (kind == "hasattr" and len(node.args) != 2):
            return node
        helper = f"_autofix_{kind}_{attr}"
        self.helpers[(kind, attr)] = helper
        args = [node.args[0]] + ([node.args[2]] if kind == "getattr" else [])
        return ast.copy_location(ast.Call(ast.Name(helper, ast.Load()), args, []), node)

    def visit_Assign(self, node):  # noqa: N802
        node = self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and len(node.targets[0].elts) == 3
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "context"
            and node.value.func.attr == "evaluate_validation"
        ):
            self.evaluation = True
            node.value = ast.copy_location(
                ast.Call(ast.Name("_autofix_metrics_tuple", ast.Load()), [node.value], []),
                node.value,
            )
        return node


def _is_shape_one_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int" and len(node.args) == 1:
        node = node.args[0]
    if isinstance(node, ast.Subscript):
        val = node.value
        sl = node.slice
        if isinstance(val, ast.Attribute) and val.attr == "shape":
            if isinstance(sl, ast.Constant) and sl.value == 1:
                return True
    return False


def _helper(kind: str, attr: str, name: str) -> ast.FunctionDef:
    args = "obj, default" if kind == "getattr" else "obj"
    success = f"return obj.{attr}" if kind == "getattr" else f"obj.{attr}"
    failure = "return default" if kind == "getattr" else "return False"
    tail = "" if kind == "getattr" else "\n    return True"
    return ast.parse(
        f"def {name}({args}):\n    try:\n        {success}\n"
        f"    except AttributeError:\n        {failure}{tail}\n"
    ).body[0]


def _insert_helpers(module: ast.Module, helpers: list[ast.stmt]) -> None:
    existing = {n.name for n in module.body if isinstance(n, ast.FunctionDef)}
    helpers = [h for h in helpers if h.name not in existing]
    index = int(bool(module.body and isinstance(module.body[0], ast.Expr)
                     and isinstance(module.body[0].value, ast.Constant)
                     and isinstance(module.body[0].value.value, str)))
    while index < len(module.body) and isinstance(module.body[index], (ast.Import, ast.ImportFrom)):
        index += 1
    module.body[index:index] = helpers


def _field_dimension(source: str) -> str:
    if ("context.field_dimension" not in source and "_field_dim" not in source) or "def run(" not in source or "_fd = context.field_dimension" in source:
        return source
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("def run("):
            lines[index + 1:index + 1] = [
                "    _fd = context.field_dimension",
                "    _field_dim = sum(_fd) if isinstance(_fd, list) else int(_fd)",
            ]
            prefix = "\n".join(lines[:index + 2])
            suffix = "\n".join(lines[index + 2:]).replace("context.field_dimension", "_field_dim")
            return prefix + "\n" + suffix + "\n"
    return source


def fix_candidate_source(source: str) -> str:
    module = ast.parse(source)
    # Strip any erroneously embedded unittest classes or main blocks from candidate.py
    kept_body = []
    for stmt in module.body:
        if isinstance(stmt, ast.ClassDef):
            if stmt.name.startswith("Test") or any(
                (isinstance(b, ast.Name) and b.id == "TestCase")
                or (isinstance(b, ast.Attribute) and b.attr == "TestCase")
                for b in stmt.bases
            ):
                continue
        elif isinstance(stmt, ast.If):
            if "unittest" in ast.dump(stmt):
                continue
        kept_body.append(stmt)
    module.body = kept_body

    transform = _Transformer()
    module = transform.visit(module)
    helpers = [_helper(kind, attr, name) for (kind, attr), name in transform.helpers.items()]
    if transform.evaluation:
        helpers.append(ast.parse(
            "def _autofix_primary(result):\n"
            "    return float(result['primary']) if isinstance(result, dict) else float(result)\n"
        ).body[0])
        helpers.append(ast.parse(
            "def _autofix_metrics_tuple(result):\n"
            "    if isinstance(result, dict):\n"
            "        return (float(result['primary']), float(result['GAUC']), float(result['nDCG@5']))\n"
            "    return result\n"
        ).body[0])
    _insert_helpers(module, helpers)
    ast.fix_missing_locations(module)
    return _field_dimension(ast.unparse(module) + "\n")


def fix_test_source(source: str) -> str:
    module = ast.parse(source)
    allowed = ALLOWED_IMPORTS | TEST_ONLY_IMPORTS
    kept = []
    for node in module.body:
        if isinstance(node, ast.Import):
            node.names = [a for a in node.names if is_allowed_import(a.name, allowed)]
            if not node.names:
                continue
        elif isinstance(node, ast.ImportFrom) and (node.level or not is_allowed_import(node.module or "", allowed)):
            continue
        kept.append(node)
    module.body = kept
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"
