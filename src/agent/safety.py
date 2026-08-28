from __future__ import annotations

import ast
import re
from pathlib import Path


ALLOWED_IMPORTS = {
    "numpy",
    "collections",
    "math",
    "time",
    "typing",
    "dataclasses",
    "src.models.fm_core",
    "src.models.sampling",
    "src.experiments.contracts",
}
TEST_ONLY_IMPORTS = {"unittest", "candidate"}
FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "dir",
    "globals",
    "locals",
}
FORBIDDEN_ATTRIBUTES = {
    "system",
    "popen",
    "call",
    "check_output",
    "remove",
    "unlink",
    "rmtree",
    "rename",
    "replace",
    "write_text",
    "write_bytes",
    "read_text",
    "read_bytes",
    "load",
    "save",
    "savez",
    "savez_compressed",
    "fromfile",
    "tofile",
    "memmap",
    "load_library",
    "loadtxt",
    "genfromtxt",
}
FORBIDDEN_TEXT = {
    "data/judge",
    "data\\judge",
    "test_truth",
    "ground_truth",
    "official_evaluate",
    "src.evaluation",
    "kuairand-starter-kit",
    "subprocess",
    "socket",
    "requests",
    "urllib",
}


class SafetyViolation(ValueError):
    pass


def validate_identifier(value: str, label: str = "identifier") -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise SafetyViolation(f"Unsafe {label}: {value!r}")
    return value


def validate_source(source: str, *, test_file: bool = False) -> None:
    lowered = source.lower()
    for forbidden in FORBIDDEN_TEXT:
        if forbidden.lower() in lowered:
            raise SafetyViolation(f"Generated source contains forbidden reference: {forbidden}")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SafetyViolation(f"Generated source is not valid Python: {exc}") from exc

    allowed = ALLOWED_IMPORTS | (TEST_ONLY_IMPORTS if test_file else set())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed:
                    raise SafetyViolation(f"Import is not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in allowed:
                raise SafetyViolation(f"Import is not allowed: {module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise SafetyViolation(f"Call is not allowed: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRIBUTES:
                raise SafetyViolation(f"Attribute call is not allowed: {node.func.attr}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SafetyViolation(f"Dunder attribute access is not allowed: {node.attr}")


def validate_family_contract(source: str, family: str) -> None:
    tree = ast.parse(source)
    required = {
        "bpr": "sample_bpr_pairs",
        "group_softmax": "sample_softmax_groups",
    }.get(family)
    if required is None:
        raise SafetyViolation(f"Unsupported candidate family: {family}")
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    if required not in called:
        raise SafetyViolation(
            f"{family} candidate must call the trusted same-user sampler {required}()."
        )


def contained_path(root: Path, *parts: str) -> Path:
    target = root.joinpath(*parts).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise SafetyViolation(f"Path escapes generated root: {target}")
    return target
