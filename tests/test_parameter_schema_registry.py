"""The structured-output parameter schema must be derived from the family registry.

Before this, `llm.PARAMETER_PROPERTIES` was a hard-coded nine with
`additionalProperties: false`, so a knob registered in `families.py` -- `history_features`'
`use_*`/`scheme`/`smoothing`, `multi_task`'s `aux_weight` -- could not appear in a model
response at all and those families silently collapsed to their frozen `defaults`.

The trap this file exists to keep closed: `required` used to be
`list(PARAMETER_PROPERTIES)`. Widening `properties` without splitting `required` would make
every derived key mandatory and break *every* existing nine-key proposal, for every family.
"""

from __future__ import annotations

import unittest

from jsonschema import Draft202012Validator, ValidationError, validate

from src.agent import llm
from src.agent.families import FAMILIES
from src.agent.policy import sanitize_parameters


#: The nine keys the schema has always carried, and the only ones a proposal must supply.
BASE_KEYS = (
    "seed",
    "k",
    "learning_rate",
    "epochs",
    "batch_size",
    "patience",
    "negatives_per_positive",
    "negatives_per_group",
    "temperature",
)

#: A nine-key proposal exactly as the model has always been able to emit one.
BASE_PROPOSAL = {
    "seed": 0,
    "k": 16,
    "learning_rate": 0.001,
    "epochs": 40,
    "batch_size": 2048,
    "patience": 4,
    "negatives_per_positive": 1,
    "negatives_per_group": None,
    "temperature": None,
}


class ParameterSchemaDerivationTests(unittest.TestCase):
    def test_schema_carries_every_grid_key_of_every_family(self):
        """The registry is the authority: register a knob, the model can emit it."""
        properties = llm.PARAMETER_SCHEMA["properties"]
        for name, family in FAMILIES.items():
            for key in family.grid:
                with self.subTest(family=name, parameter=key):
                    self.assertIn(key, properties)

    def test_the_original_nine_are_required_and_only_those(self):
        required = llm.PARAMETER_SCHEMA["required"]
        self.assertEqual(list(required), list(BASE_KEYS))
        # Not a set comparison: a duplicate or a reordering would still be a change to a
        # contract the model is prompt-cached against.
        self.assertEqual(len(required), len(BASE_KEYS))
        derived = set(llm.PARAMETER_SCHEMA["properties"]) - set(BASE_KEYS)
        self.assertTrue(derived, "no family-specific key was derived from the registry")
        for key in sorted(derived):
            with self.subTest(derived=key):
                self.assertNotIn(key, required)

    def test_the_schema_is_still_closed_and_still_valid(self):
        self.assertFalse(llm.PARAMETER_SCHEMA["additionalProperties"])
        self.assertEqual(llm.PARAMETER_SCHEMA["type"], "object")
        Draft202012Validator.check_schema(llm.PARAMETER_SCHEMA)
        # Both roles that carry parameters route through the one constant.
        for schema_name in ("research_decision", "candidate_manifest"):
            with self.subTest(schema=schema_name):
                self.assertIs(
                    llm.SCHEMAS[schema_name]["properties"]["parameters"],
                    llm.PARAMETER_SCHEMA,
                )

    def test_derived_types_are_inferred_from_the_grid_shape(self):
        properties = llm.PARAMETER_SCHEMA["properties"]
        # `range` -> integer, tuple of floats -> number, tuple of bools -> boolean,
        # tuple of strings -> string.
        self.assertEqual(properties["l2"], {"type": "number"})
        self.assertEqual(properties["aux_weight"], {"type": "number"})
        self.assertEqual(properties["smoothing"], {"type": "number"})
        self.assertEqual(properties["scheme"], {"type": "string"})
        self.assertEqual(properties["use_user_author"], {"type": "boolean"})
        self.assertEqual(properties["use_is_click"], {"type": "boolean"})
        # `bool` is a subclass of `int` in Python, so a naive isinstance order would type
        # every toggle as an integer and let the model emit `use_is_click: 3`.
        self.assertEqual(llm._grid_property((True, False)), {"type": "boolean"})
        self.assertEqual(llm._grid_property(range(0, 1000)), {"type": "integer"})
        self.assertEqual(llm._grid_property((8, 16, 32, 64)), {"type": "integer"})

    def test_derived_types_match_the_values_the_grid_actually_allows(self):
        """A derived type that rejects its own grid would cost an iteration on every use."""
        validators = {
            "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
            "number": lambda value: isinstance(value, (int, float))
            and not isinstance(value, bool),
            "boolean": lambda value: isinstance(value, bool),
            "string": lambda value: isinstance(value, str),
        }
        properties = llm.PARAMETER_SCHEMA["properties"]
        for name, family in FAMILIES.items():
            for key, allowed in family.grid.items():
                if key in BASE_KEYS:
                    continue  # the base nine keep their own hand-written entries
                declared = properties[key].get("type")
                values = (0, 999) if isinstance(allowed, range) else tuple(allowed)
                for value in values:
                    with self.subTest(family=name, parameter=key, value=value):
                        self.assertTrue(validators[declared](value))


class ProposalValidationTests(unittest.TestCase):
    def test_an_unchanged_nine_key_proposal_still_validates(self):
        validate(instance=BASE_PROPOSAL, schema=llm.PARAMETER_SCHEMA)

    def test_a_proposal_carrying_family_specific_extras_validates(self):
        """The whole point: `history_features` and `multi_task` can now be varied."""
        history = {
            **BASE_PROPOSAL,
            "epochs": 20,
            "scheme": "prior_days",
            "smoothing": 100.0,
            "use_user_author": False,
            "use_recency": True,
        }
        validate(instance=history, schema=llm.PARAMETER_SCHEMA)
        multi_task = {**BASE_PROPOSAL, "aux_weight": 0.3, "use_is_like": True}
        validate(instance=multi_task, schema=llm.PARAMETER_SCHEMA)
        capacity = {**BASE_PROPOSAL, "k": 64, "l2": 1e-2, "learning_rate": 0.005}
        validate(instance=capacity, schema=llm.PARAMETER_SCHEMA)

    def test_a_proposal_missing_one_of_the_nine_is_still_rejected(self):
        for key in BASE_KEYS:
            with self.subTest(missing=key):
                incomplete = {k: v for k, v in BASE_PROPOSAL.items() if k != key}
                with self.assertRaises(ValidationError):
                    validate(instance=incomplete, schema=llm.PARAMETER_SCHEMA)

    def test_a_knob_no_family_registers_is_still_rejected(self):
        self.assertNotIn(
            "use_author_affinity",
            {key for family in FAMILIES.values() for key in family.grid},
        )
        with self.assertRaises(ValidationError):
            validate(
                instance={**BASE_PROPOSAL, "use_author_affinity": True},
                schema=llm.PARAMETER_SCHEMA,
            )

    def test_a_full_research_decision_with_extras_validates(self):
        decision = {
            "hypothesis_id": "h_1",
            "family": "multi_task",
            "action": "explore",
            "hypothesis": "auxiliary click signal helps long_view ranking",
            "rationale": "method card",
            "parameters": {**BASE_PROPOSAL, "aux_weight": 0.1, "use_is_like": True},
            "evidence": [],
            "needs_web_search": False,
            "parent_experiment": None,
        }
        validate(instance=decision, schema=llm.SCHEMAS["research_decision"])


class GridDefaultTests(unittest.TestCase):
    """The widened capacity axes must leave an unchanged proposal byte-identical."""

    LOSS_FAMILIES = ("bpr", "group_softmax")

    def test_capacity_defaults_are_todays_values(self):
        for name in self.LOSS_FAMILIES:
            with self.subTest(family=name):
                defaults = FAMILIES[name].defaults
                self.assertEqual(defaults["k"], 16)
                self.assertEqual(defaults["l2"], 1e-6)
                self.assertEqual(defaults["learning_rate"], 0.001)
                self.assertEqual(defaults["epochs"], 40)
                self.assertEqual(defaults["patience"], 4)
                self.assertEqual(defaults["seed"], 0)
                self.assertEqual(defaults["batch_size"], 2048)

    def test_the_widened_axes_are_supersets_of_todays_single_points(self):
        for name in self.LOSS_FAMILIES:
            grid = FAMILIES[name].grid
            with self.subTest(family=name):
                self.assertEqual(grid["k"], (8, 16, 32, 64))
                # Magnitudes track W1b's decoupled decay (per-step shrink = lr*l2):
                # the whole 1e-6..1e-4 band is <=1e-7 per step, i.e. off. `1e-6` stays
                # only because it is the default, and a default must be a grid member.
                self.assertEqual(grid["l2"], (0.0, 1e-6, 1e-4, 1e-3, 1e-2))
                self.assertEqual(max(grid["l2"]), 1e-2)
                # Widening must never *remove* a point the method cards advertise.
                for rate in (0.0003, 0.0005, 0.001):
                    self.assertIn(rate, grid["learning_rate"])
                self.assertEqual(max(grid["learning_rate"]), 0.005)

    def test_pair_weighting_is_registered_and_derived_into_the_schema(self):
        """C6c: a new grid key must become proposable without any edit to `llm.py`."""
        grid = FAMILIES["bpr"].grid
        self.assertEqual(grid["pair_weighting"], ("none", "delta_ndcg"))
        # `none` is today's plain BPR gradient, so an unchanged proposal is unaffected.
        self.assertEqual(FAMILIES["bpr"].defaults["pair_weighting"], "none")
        # Derived, not hand-written: a tuple of strings types as a string, and the key stays
        # optional so the other three families never have to emit it.
        self.assertEqual(
            llm.PARAMETER_SCHEMA["properties"]["pair_weighting"], {"type": "string"}
        )
        self.assertNotIn("pair_weighting", llm.PARAMETER_SCHEMA["required"])
        for value in grid["pair_weighting"]:
            with self.subTest(value=value):
                validate(
                    instance={**BASE_PROPOSAL, "pair_weighting": value},
                    schema=llm.PARAMETER_SCHEMA,
                )
                sanitised = sanitize_parameters("bpr", {"pair_weighting": value})
                self.assertEqual(sanitised["pair_weighting"], value)
        # The grid, not the schema, is what rejects an invented weighting.
        with self.assertRaises(ValueError) as rejected:
            sanitize_parameters("bpr", {"pair_weighting": "lambdamart"})
        self.assertIn("pair_weighting", str(rejected.exception))

    def test_leave_one_out_is_no_longer_proposable(self):
        """C6c-bis: LOO is the measurably worse target-statistic scheme (arXiv:1706.09516).

        It stays implemented in `features.py` for direct callers -- only the *search space*
        loses it, so the agent cannot spend an exploit iteration on a dead axis.
        """
        family = FAMILIES["history_features"]
        self.assertEqual(family.grid["scheme"], ("prior_days",))
        self.assertEqual(family.defaults["scheme"], "prior_days")
        self.assertNotIn(
            "leave_one_out",
            {value for grid in FAMILIES.values() for value in grid.grid.get("scheme", ())},
        )
        with self.assertRaises(ValueError) as rejected:
            sanitize_parameters("history_features", {"scheme": "leave_one_out"})
        self.assertIn("scheme", str(rejected.exception))

    def test_every_grid_key_has_a_default_inside_its_own_grid(self):
        for name, family in FAMILIES.items():
            for key, allowed in family.grid.items():
                with self.subTest(family=name, parameter=key):
                    self.assertIn(key, family.defaults)
                    self.assertIn(family.defaults[key], allowed)

    def test_an_empty_proposal_still_sanitises_to_exactly_the_defaults(self):
        for name in self.LOSS_FAMILIES:
            with self.subTest(family=name):
                self.assertEqual(sanitize_parameters(name, {}), FAMILIES[name].defaults)

    def test_the_sanitiser_accepts_every_widened_capacity_value(self):
        for name in self.LOSS_FAMILIES:
            defaults = FAMILIES[name].defaults
            for key in ("k", "l2", "learning_rate"):
                for value in FAMILIES[name].grid[key]:
                    with self.subTest(family=name, parameter=key, value=value):
                        sanitised = sanitize_parameters(name, {**defaults, key: value})
                        self.assertEqual(sanitised[key], value)

    def test_the_sanitiser_still_rejects_a_value_off_the_widened_grid(self):
        for name in self.LOSS_FAMILIES:
            for key, off_grid in (("k", 128), ("l2", 0.5), ("learning_rate", 0.5)):
                # 0.5 is off both the old and the widened l2 tuple.
                with self.subTest(family=name, parameter=key):
                    with self.assertRaises(ValueError) as rejected:
                        sanitize_parameters(name, {**FAMILIES[name].defaults, key: off_grid})
                    self.assertIn(key, str(rejected.exception))


if __name__ == "__main__":
    unittest.main()
