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
import builtins
import re
from pathlib import Path

from .families import required_call_groups


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


# Everything in ``builtins`` a training loop legitimately needs: pure builders, iterators,
# numerics, the class machinery and the exception types. Every name in FORBIDDEN_CALLS is
# excluded by construction -- see the assertion below, which fails the import if that drifts.
SAFE_BUILTIN_NAMES = frozenset(
    {
        # constructors and containers
        "bool", "bytearray", "bytes", "complex", "dict", "float", "frozenset", "int",
        "list", "memoryview", "object", "set", "slice", "str", "tuple",
        # iteration and functional helpers
        "all", "any", "enumerate", "filter", "iter", "len", "map", "next", "range",
        "reversed", "sorted", "sum", "zip",
        # numerics and formatting
        "abs", "ascii", "bin", "chr", "divmod", "format", "hex", "max", "min", "oct",
        "ord", "pow", "repr", "round",
        # introspection that cannot reach the filesystem
        "callable", "hasattr", "hash", "id", "isinstance", "issubclass", "type",
        # class machinery
        "classmethod", "property", "staticmethod", "super",
        # diagnostics
        "print",
        # sentinels
        "Ellipsis", "NotImplemented",
        # exception types a candidate may raise or catch
        "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
        "Exception", "FloatingPointError", "ImportError", "IndexError", "KeyError",
        "LookupError", "MemoryError", "NameError", "NotImplementedError", "OSError",
        "OverflowError", "RecursionError", "RuntimeError", "StopIteration",
        "TypeError", "UnboundLocalError", "ValueError", "ZeroDivisionError",
    }
)
assert not (SAFE_BUILTIN_NAMES & FORBIDDEN_CALLS), "safe builtins must exclude FORBIDDEN_CALLS"


def _guarded_import(*, test_file: bool = False):
    """Return an ``__import__`` replacement enforcing the same allowlist as ``validate_source``."""
    allowed = ALLOWED_IMPORTS | (TEST_ONLY_IMPORTS if test_file else set())

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            raise SafetyViolation("Relative imports are not allowed.")
        if not is_allowed_import(name, allowed):
            raise SafetyViolation(f"Import is not allowed: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    return guarded


def restricted_builtins(*, test_file: bool = False) -> dict[str, object]:
    """A ``__builtins__`` mapping for executed candidate code.

    Defence in depth behind ``validate_source``: even if a bypass were found in the AST rules,
    the executed module cannot reach ``open`` (absent -> NameError) or import outside the
    allowlist (guarded ``__import__`` -> SafetyViolation).

    Scope, deliberately narrow: this covers the **training** run, where
    ``run_candidate.py::_load_candidate`` execs the candidate module and can pre-set the key.
    It does not cover the unit-test subprocess, which imports ``candidate.py`` through the
    normal import system (``candidate_runner.py``), nor modules the candidate imports -- those
    keep their own real builtins. T1's AST rules stay the primary defence.

    Precondition: the executing namespace must carry ``__name__`` -- a class body reads it to
    fill ``__module__``. ``importlib.util.module_from_spec`` sets it, so the real call site is
    already correct; a bare ``exec`` namespace has to supply it.
    """
    mapping: dict[str, object] = {
        name: getattr(builtins, name) for name in sorted(SAFE_BUILTIN_NAMES)
    }
    # Without __build_class__ every `class` statement in the candidate fails.
    mapping["__build_class__"] = builtins.__build_class__
    mapping["__import__"] = _guarded_import(test_file=test_file)
    return mapping


def validate_family_contract(source: str, family: str) -> None:
    """Require the trusted helpers the registry declares for ``family``.

    The requirement is a tuple of one-of groups, so a family may demand "one of the two
    samplers *and* build_features" without this function growing a second literal list.
    """
    try:
        groups = required_call_groups(family)
    except KeyError:
        raise SafetyViolation(f"Unsupported candidate family: {family}") from None

    tree = ast.parse(source)
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)

    for group in groups:
        if any(name in called for name in group):
            continue
        if len(group) == 1:
            raise SafetyViolation(
                f"{family} candidate must call the trusted same-user sampler {group[0]}()."
            )
        rendered = ", ".join(f"{name}()" for name in group)
        raise SafetyViolation(
            f"{family} candidate must call at least one of: {rendered}."
        )


def contained_path(root: Path, *parts: str) -> Path:
    target = root.joinpath(*parts).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise SafetyViolation(f"Path escapes generated root: {target}")
    return target
