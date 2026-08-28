from __future__ import annotations

import unittest

import pytest

from src.agent import errors
from src.agent.errors import (
    IncompleteResponse,
    LLMError,
    RoleOutputInvalid,
    TokenBudgetExceeded,
)


class TypedErrorTests(unittest.TestCase):
    def test_every_error_class_is_importable(self):
        for name in ("LLMError", "TokenBudgetExceeded", "RoleOutputInvalid", "IncompleteResponse"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(errors, name))
                self.assertTrue(issubclass(getattr(errors, name), Exception))

    def test_base_error_subclasses_runtime_error(self):
        self.assertTrue(issubclass(LLMError, RuntimeError))

    def test_specific_errors_subclass_llm_error(self):
        for cls in (TokenBudgetExceeded, RoleOutputInvalid, IncompleteResponse):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, LLMError))


@pytest.mark.slow
def test_slow_marker_is_registered():
    assert True


if __name__ == "__main__":
    unittest.main()
