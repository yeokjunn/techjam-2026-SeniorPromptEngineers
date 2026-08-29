"""Static and dynamic guards for LLM-generated candidate code.

``validate_source`` is the only barrier between generated code and the raw logs, so it is
deliberately conservative. Two rules keep it honest:

* **An attribute is banned only if a capability with that name is reachable from an allowed
  module.** With ``os``/``pathlib``/``subprocess`` unimportable, the only receivers for
  ``.replace``/``.rename`` are ``str``, ``bytes`` and numpy arrays, where those methods are pure --
  so banning them only burns Debugger repairs on false positives. Everything left in
  ``FORBIDDEN_ATTRIBUTES`` names a real numpy or OS capability, and is rejected on *access*, not
  just on call, so aliasing (``f = np.load``) cannot defeat it.
* **The static and dynamic import rules share one predicate.** ``is_allowed_import`` backs both the
  AST walk here and the guarded ``__import__`` handed to executed candidates, so the two cannot
  drift apart.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ALLOWED_IMPORTS = {
    "__future__",
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
# ``__name__`` is the one dunder a candidate legitimately reads: without it every generated test
# carrying ``if __name__ == "__main__":`` would be rejected.
ALLOWED_DUNDER_NAMES = {"__name__"}
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
    "check_output",
    "remove",
    "unlink",
    "rmtree",
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
# The hidden test labels live in the 20220429-20220508 range inside the standard log, so the raw
# dataset paths are tripwires in their own right. Matching is case-insensitive.
FORBIDDEN_TEXT = {
    "test_truth",
    "ground_truth",
    "official_evaluate",
    "src.evaluation",
    "log_standard",
    "log_random",
    "kuairand",
    ".csv",
    "/data/",
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


def is_allowed_import(name: str, allowed: set[str]) -> bool:
    """Allow an allowlisted module and any submodule of one (``numpy`` covers ``numpy.random``)."""
    return name in allowed or any(name.startswith(prefix + ".") for prefix in allowed)


def _is_forbidden_dunder(name: str) -> bool:
    return name.startswith("__") and name not in ALLOWED_DUNDER_NAMES


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
                if not is_allowed_import(alias.name, allowed):
                    raise SafetyViolation(f"Import is not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise SafetyViolation("Relative imports are not allowed.")
            module = node.module or ""
            if not is_allowed_import(module, allowed):
                raise SafetyViolation(f"Import is not allowed: {module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise SafetyViolation(f"Call is not allowed: {node.func.id}")
        elif isinstance(node, ast.Attribute):
            # Access, not just call: ``f = np.load`` must fail as hard as ``np.load(...)``.
            if node.attr in FORBIDDEN_ATTRIBUTES:
                raise SafetyViolation(f"Attribute access is not allowed: {node.attr}")
            if node.attr.startswith("__"):
                raise SafetyViolation(f"Dunder attribute access is not allowed: {node.attr}")
        elif isinstance(node, ast.Name):
            # ``__builtins__`` is a Name, so ``__builtins__['open']`` never reaches an Attribute.
            if _is_forbidden_dunder(node.id):
                raise SafetyViolation(f"Dunder name is not allowed: {node.id}")
        elif isinstance(node, ast.Subscript):
            # Redundant while no dunder is allowlisted; keeps the hole shut if one ever is.
            target = node.value
            if isinstance(target, ast.Name) and target.id.startswith("__"):
                raise SafetyViolation(f"Subscript of a dunder name is not allowed: {target.id}")


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
