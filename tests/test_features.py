"""Leakage and layout guarantees for the trusted history feature builder (review I8)."""

from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path

import numpy as np

from src.agent.families import AUX_HEADS as fam_aux_heads
from src.agent.families import FAMILIES, coverage_families
from src.agent.families import HISTORY_GROUPS as fam_history_groups
from src.agent.policy import sanitize_parameters
from src.models.features import (
    AUX_HEADS,
    GROUPS,
    SLOTS_PER_GROUP,
    UNKNOWN_SLOT,
    aux_dimension,
    build_aux_labels,
    build_features,
    enabled_groups,
    feature_dimension,
)


def row(date: int, user: str, video: str, author: str, tab: str, duration: float, label):
    """A kit-shaped row: (date, user_id, video_id, author_id, tab, duration_ms, long_view)."""
    return (date, user, video, author, tab, duration, label)


def upload_dates(**mapping: str) -> dict[str, int]:
    return {video: dt.date.fromisoformat(day).toordinal() for video, day in mapping.items()}


class ExplosiveLabel:
    """Raises on any use, so a test fails loudly if a non-train label is ever read."""

    def __bool__(self):
        raise AssertionError("a non-train label was read")

    def __float__(self):
        raise AssertionError("a non-train label was read")

    def __int__(self):
        raise AssertionError("a non-train label was read")

    def __eq__(self, other):
        raise AssertionError("a non-train label was read")

    def __gt__(self, other):
        raise AssertionError("a non-train label was read")

    def __hash__(self):
        return 0

    def __repr__(self):
        return "<ExplosiveLabel>"


class FeatureBuilderTests(unittest.TestCase):
    TRAIN = [
        row(20220408, "u1", "v1", "a1", "1", 1000.0, 0),
        row(20220408, "u1", "v2", "a1", "1", 2000.0, 1),
        row(20220409, "u1", "v3", "a2", "2", 3000.0, 1),
        row(20220410, "u2", "v1", "a1", "1", 1500.0, 0),
        row(20220410, "u1", "v4", "a1", "1", 2500.0, 0),
    ]
    UPLOADS = upload_dates(v1="2022-04-01", v2="2022-04-02", v3="2022-04-03", v4="2022-04-04")

    def spec(self, split="train", *, valid=None, test=None, **overrides):
        history = {"train": self.TRAIN}
        if valid is not None:
            history["valid"] = valid
        if test is not None:
            history["test"] = test
        base = {
            "split": split,
            "field_offset": 100,
            "history_rows": history,
            "video_upload_dates": self.UPLOADS,
        }
        base.update(overrides)
        return base

    def build(self, rows, spec):
        return build_features(np.zeros((len(rows), 5)), spec)

    def slot(self, features, group, spec):
        """Recover the within-group slot of a column, undoing offset and group base."""
        groups = enabled_groups(spec)
        column = groups.index(group)
        return features[:, column] - spec["field_offset"] - column * SLOTS_PER_GROUP

    # -- leakage ------------------------------------------------------------------------

    def test_train_features_use_only_strictly_earlier_days(self):
        spec = self.spec()
        features = self.build(self.TRAIN, spec)
        rates = self.slot(features, "user_rate", spec)
        # u1's first day (rows 0-1) and u2's first appearance (row 3) have no prior history.
        self.assertEqual(rates[0], UNKNOWN_SLOT)
        self.assertEqual(rates[1], UNKNOWN_SLOT)
        self.assertEqual(rates[3], UNKNOWN_SLOT)
        # By day two u1 has a populated bucket.
        self.assertNotEqual(rates[2], UNKNOWN_SLOT)
        self.assertNotEqual(rates[4], UNKNOWN_SLOT)

    def test_same_day_rows_cannot_see_each_others_labels(self):
        """Row 1 is u1's only day-one positive; row 0 must not benefit from it."""
        spec = self.spec()
        rates = self.slot(self.build(self.TRAIN, spec), "user_rate", spec)
        self.assertEqual(rates[0], UNKNOWN_SLOT)

    def test_valid_rows_use_all_of_train(self):
        valid = [row(20220422, "u1", "v1", "a1", "1", 1000.0, 0)]
        spec = self.spec("valid", valid=valid)
        rates = self.slot(self.build(valid, spec), "user_rate", spec)
        # u1 is seen throughout train, so a valid row is never "no history".
        self.assertNotEqual(rates[0], UNKNOWN_SLOT)

    def test_labels_of_non_train_rows_are_never_read(self):
        valid = [
            row(20220422, "u1", "v1", "a1", "1", 1000.0, ExplosiveLabel()),
            row(20220423, "u2", "v2", "a1", "1", 2000.0, ExplosiveLabel()),
        ]
        spec = self.spec("valid", valid=valid)
        features = self.build(valid, spec)  # must not raise
        self.assertEqual(features.shape, (2, len(GROUPS)))

    def test_test_split_labels_are_never_read(self):
        test = [row(20220429, "u1", "v1", "a1", "1", 1000.0, ExplosiveLabel())]
        spec = self.spec("test", test=test)
        self.assertEqual(self.build(test, spec).shape, (1, len(GROUPS)))

    # -- layout -------------------------------------------------------------------------

    def test_output_is_deterministic_and_within_range(self):
        spec = self.spec()
        first = self.build(self.TRAIN, spec)
        second = self.build(self.TRAIN, spec)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.int32)
        low = spec["field_offset"]
        self.assertTrue((first >= low).all())
        self.assertTrue((first < low + feature_dimension(spec)).all())

    def test_realistic_field_offset_does_not_overflow(self):
        """KuaiRand-Pure's field_dimension is 40260, well past int16."""
        spec = self.spec()
        spec["field_offset"] = 40260
        features = self.build(self.TRAIN, spec)
        self.assertEqual(features.dtype, np.int32)
        self.assertTrue((features >= 40260).all())
        self.assertTrue((features < 40260 + feature_dimension(spec)).all())

    def test_feature_dimension_matches_enabled_groups(self):
        self.assertEqual(feature_dimension({}), SLOTS_PER_GROUP * len(GROUPS))
        for disabled in range(len(GROUPS) + 1):
            spec = {f"use_{name}": False for name in GROUPS[:disabled]}
            with self.subTest(disabled=disabled):
                self.assertEqual(
                    feature_dimension(spec), SLOTS_PER_GROUP * (len(GROUPS) - disabled)
                )

    def test_disabled_groups_leave_the_remaining_columns_contiguous(self):
        spec = self.spec(use_user_rate=False, use_recency=False)
        features = self.build(self.TRAIN, spec)
        self.assertEqual(features.shape[1], len(GROUPS) - 2)
        self.assertTrue((features < spec["field_offset"] + feature_dimension(spec)).all())

    def test_unknown_keys_fall_in_the_reserved_slot(self):
        valid = [row(20220422, "unseen", "unseen", "unseen", "unseen", 99999.0, 0)]
        spec = self.spec("valid", valid=valid)
        features = self.build(valid, spec)
        for group in GROUPS:
            with self.subTest(group=group):
                self.assertEqual(self.slot(features, group, spec)[0], UNKNOWN_SLOT)

    def test_missing_upload_date_is_unknown_not_a_value_bucket(self):
        valid = [row(20220422, "u1", "no_upload_record", "a1", "1", 1000.0, 0)]
        spec = self.spec("valid", valid=valid)
        self.assertEqual(self.slot(self.build(valid, spec), "video_age", spec)[0], UNKNOWN_SLOT)

    # -- contract -----------------------------------------------------------------------

    def test_schemes_differ_on_train_rows(self):
        prior = self.build(self.TRAIN, self.spec(scheme="prior_days"))
        loo = self.build(self.TRAIN, self.spec(scheme="leave_one_out"))
        self.assertFalse(np.array_equal(prior, loo))

    def test_row_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_features(np.zeros((len(self.TRAIN) + 1, 5)), self.spec())

    def test_invalid_spec_values_are_rejected(self):
        for overrides in ({"split": "holdout"}, {"scheme": "magic"}, {"smoothing": 0.0}):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                spec = self.spec()
                spec.update(overrides)
                self.build(self.TRAIN, spec)

    def test_smoothing_pulls_sparse_keys_toward_the_prior(self):
        """A heavier prior must not change the layout, only the values inside it."""
        light = self.build(self.TRAIN, self.spec(smoothing=1.0))
        heavy = self.build(self.TRAIN, self.spec(smoothing=1000.0))
        self.assertEqual(light.shape, heavy.shape)
        self.assertEqual(light.dtype, heavy.dtype)

class AuxiliaryLabelTests(unittest.TestCase):
    """The multi_task family's auxiliary targets: (is_click, is_like, is_follow, is_comment, is_forward, play_time_ms) per train row."""

    AUX = [
        (1, 0, 0, 0, 0, 500.0),
        (0, 0, 0, 0, 0, 0.0),
        (1, 1, 1, 0, 1, 120000.0),
        (0, 0, 0, 0, 0, 250.0),
        (1, 0, 0, 1, 0, 9000.0),
    ]

    def spec(self, split="train", **overrides):
        base = {"split": split, "aux_rows": self.AUX}
        base.update(overrides)
        return base

    def build(self, spec, rows=None):
        return build_aux_labels(np.zeros((len(rows or self.AUX), 5)), spec)

    def test_aux_labels_are_train_only_and_finite(self):
        targets = self.build(self.spec())
        self.assertEqual(targets.shape, (len(self.AUX), len(AUX_HEADS)))
        self.assertEqual(targets.dtype, np.float32)
        self.assertTrue(np.isfinite(targets).all())
        self.assertTrue(((targets >= 0.0) & (targets <= 1.0)).all())
        for split in ("valid", "test"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.build(self.spec(split))

    def test_binary_heads_are_passed_through_unchanged(self):
        targets = self.build(self.spec())
        np.testing.assert_array_equal(targets[:, 0], [1.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(targets[:, 1], [0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(targets[:, 2], [0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(targets[:, 3], [0.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(targets[:, 4], [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_play_time_is_log_compressed_and_min_max_scaled(self):
        play = self.build(self.spec())[:, 5]
        self.assertAlmostEqual(float(play.min()), 0.0, places=6)
        self.assertAlmostEqual(float(play.max()), 1.0, places=6)
        # log1p compression: the 120000ms row must not dwarf everything else linearly.
        self.assertGreater(float(play[4]), 120000.0 / 120000.0 * 0.5)

    def test_constant_play_time_does_not_divide_by_zero(self):
        spec = self.spec(aux_rows=[(0, 0, 0, 0, 0, 42.0), (1, 0, 0, 0, 0, 42.0)])
        play = build_aux_labels(np.zeros((2, 5)), spec)[:, 5]
        self.assertTrue(np.isfinite(play).all())
        np.testing.assert_array_equal(play, [0.0, 0.0])

    def test_aux_dimension_tracks_enabled_heads(self):
        self.assertEqual(aux_dimension({}), len(AUX_HEADS))
        spec = self.spec(use_is_like=False)
        self.assertEqual(aux_dimension(spec), len(AUX_HEADS) - 1)
        self.assertEqual(self.build(spec).shape[1], len(AUX_HEADS) - 1)

    def test_disabling_every_head_is_rejected(self):
        spec = self.spec(**{f"use_{head}": False for head in AUX_HEADS})
        with self.assertRaises(ValueError):
            self.build(spec)

    def test_row_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_aux_labels(np.zeros((len(self.AUX) + 1, 5)), self.spec())


class RegistryContractTests(unittest.TestCase):
    """The registry entries must match what the trusted builders actually provide."""

    REPO_ROOT = Path(__file__).resolve().parents[1]
    FEATURE_FAMILIES = ("history_features", "multi_task")

    def test_registry_entry_matches_the_builder_contract(self):
        self.assertEqual(FAMILIES["history_features"].required_calls[1], ("build_features",))
        self.assertEqual(FAMILIES["multi_task"].required_calls[1], ("build_aux_labels",))
        for name in self.FEATURE_FAMILIES:
            with self.subTest(family=name):
                entry = FAMILIES[name]
                # Either sampler is acceptable: the loss is not what these families vary.
                self.assertEqual(
                    set(entry.required_calls[0]),
                    {"sample_bpr_pairs", "sample_softmax_groups"},
                )
                self.assertTrue((self.REPO_ROOT / entry.method_card).is_file())

    def test_method_cards_carry_every_heading_of_the_reference_card(self):
        def headings(card: str) -> list[str]:
            text = (self.REPO_ROOT / "research" / "methods" / f"{card}.md").read_text(
                encoding="utf-8"
            )
            return [line for line in text.splitlines() if line.startswith("## ")]

        reference = headings("bpr")
        for name in self.FEATURE_FAMILIES:
            with self.subTest(family=name):
                self.assertEqual(headings(name), reference)

    def test_every_grid_value_is_a_tuple_or_a_range(self):
        for name, entry in FAMILIES.items():
            for key, allowed in entry.grid.items():
                with self.subTest(family=name, parameter=key):
                    self.assertIsInstance(allowed, (tuple, range))
                    self.assertTrue(len(allowed) > 0)

    def test_every_grid_key_has_a_default_inside_its_grid(self):
        """A default outside its own grid would reject the family's own fallback."""
        for name, entry in FAMILIES.items():
            for key, allowed in entry.grid.items():
                with self.subTest(family=name, parameter=key):
                    self.assertIn(key, entry.defaults)
                    self.assertIn(entry.defaults[key], allowed)

    def test_multi_task_defaults_to_low_weight_click_only(self):
        parameters = sanitize_parameters("multi_task", {})
        self.assertEqual(parameters["aux_weight"], 0.05)
        self.assertTrue(parameters["use_is_click"])
        for head in AUX_HEADS:
            if head != "is_click":
                self.assertFalse(parameters[f"use_{head}"])

    def test_registry_toggles_match_the_feature_modules_own_names(self):
        """families.py keeps literals so types.py's import stays light; pin them here."""
        self.assertEqual(fam_history_groups, GROUPS)
        self.assertEqual(fam_aux_heads, AUX_HEADS)
        for group in GROUPS:
            self.assertIn(f"use_{group}", FAMILIES["history_features"].grid)
        for head in AUX_HEADS:
            self.assertIn(f"use_{head}", FAMILIES["multi_task"].grid)

    def test_new_families_are_sanitised_without_touching_policy(self):
        """A's I-7 reads grid/defaults from the registry; nothing in policy.py names these."""
        for name in self.FEATURE_FAMILIES:
            with self.subTest(family=name):
                parameters = sanitize_parameters(name, {})
                for key, value in FAMILIES[name].defaults.items():
                    self.assertEqual(parameters[key], value)
                with self.assertRaises(ValueError):
                    sanitize_parameters(name, {"learning_rate": 0.5})

    def test_builder_brief_carries_the_real_call_signature(self):
        """Naming a helper without its signature makes the Builder guess the argument order.

        An observed DeepSeek run emitted sample_bpr_pairs(X, y, users, 1) and burned both
        Debugger repairs on `unhashable type: numpy.ndarray`.
        """
        import inspect

        from src.agent.families import builder_brief
        from src.models import sampling

        brief = builder_brief("bpr")
        expected = str(inspect.signature(sampling.sample_bpr_pairs))
        self.assertIn(f"sample_bpr_pairs{expected}", brief)
        for parameter in ("users", "labels", "rng"):
            self.assertIn(parameter, brief)

    def test_builder_brief_signatures_cover_every_required_call(self):
        from src.agent.families import FAMILIES, builder_brief, required_call_groups

        for name in FAMILIES:
            brief = builder_brief(name)
            for group in required_call_groups(name):
                for call in group:
                    with self.subTest(family=name, call=call):
                        # every mandatory helper appears with an argument list, not a bare ()
                        self.assertIn(f"{call}(", brief)
                        self.assertNotIn(f"{call}()", brief)

    def test_builder_brief_states_the_candidate_id_rule(self):
        """An observed run lost an iteration to a dot in candidate_id; the prompt now says so."""
        import re

        from src.agent.families import FAMILIES, builder_brief
        from src.agent.safety import SafetyViolation, validate_identifier

        for name in FAMILIES:
            with self.subTest(family=name):
                self.assertIn("[A-Za-z0-9_-]", builder_brief(name))
        # and the rule stated is the rule enforced
        validate_identifier("bpr_seed42_lr5e-4")
        with self.assertRaises(SafetyViolation):
            validate_identifier("gsm_1_seed42_k16_lr5e-4_neg4_T1.0")

    def test_builder_brief_states_the_sampler_return_shapes(self):
        """The 2-D negatives array was reshaped by an agent and cost an iteration."""
        from src.agent.families import FAMILIES, builder_brief

        gs = builder_brief("group_softmax")
        self.assertIn("(n_groups, negatives_per_group)", gs)
        self.assertIn("do NOT reshape", gs)
        self.assertIn("(n_pairs,)", builder_brief("bpr"))
        # the feature families call a sampler but never see bpr.md / group_softmax.md
        for name in ("history_features", "multi_task"):
            with self.subTest(family=name):
                self.assertIn("row indices", builder_brief(name))

    def test_sampler_cards_document_the_real_return_shapes(self):
        """Card text must match what the samplers actually return."""
        import numpy as np

        from src.models.sampling import sample_bpr_pairs, sample_softmax_groups

        users = ["a", "a", "a", "b", "b"]
        labels = np.array([1, 1, 0, 1, 0], dtype=np.float32)
        pos, neg = sample_bpr_pairs(users, labels, np.random.default_rng(0), 2)
        self.assertEqual(pos.ndim, 1)
        self.assertEqual(neg.shape, pos.shape)          # parallel 1-D, as bpr.md says
        gpos, gneg = sample_softmax_groups(users, labels, np.random.default_rng(0), 4)
        self.assertEqual(gneg.shape, (len(gpos), 4))     # already 2-D, as group_softmax.md says

        card = (Path(__file__).resolve().parents[1] / "research/methods/group_softmax.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("(n_groups, negatives_per_group)", card)
        self.assertIn("do not reshape it", card)

    def test_candidate_contract_matches_the_real_fmranker_api(self):
        """Agents guessed model.w0/.w/.v and crashed; the contract must track the real API."""
        from src.agent.roles import BASE_CANDIDATE_CONTRACT
        from src.models.fm_core import FMRanker

        for method in ("logits", "gradients", "apply_gradients", "predict",
                       "state_dict", "load_state_dict"):
            with self.subTest(method=method):
                self.assertTrue(hasattr(FMRanker, method))
                self.assertIn(method, BASE_CANDIDATE_CONTRACT)
        for attribute in ("V", "W", "b"):
            with self.subTest(attribute=attribute):
                self.assertIn(f"model.{attribute}", BASE_CANDIDATE_CONTRACT)
        # and the attributes that do not exist are called out by name
        model = FMRanker(8, embedding_dim=2)
        for wrong in ("w0", "v"):
            with self.subTest(wrong=wrong):
                self.assertFalse(hasattr(model, wrong))
        collapsed = " ".join(BASE_CANDIDATE_CONTRACT.split())
        self.assertIn("no model.w0, model.w or model.v", collapsed)

    def test_contract_warns_that_state_dict_returns_copies(self):
        """Iteration 2 hand-rolled a restore: state_dict() copies, and b is 0-dimensional."""
        import numpy as np

        from src.agent.roles import BASE_CANDIDATE_CONTRACT
        from src.models.fm_core import FMRanker

        model = FMRanker(8, embedding_dim=2)
        state = model.state_dict()
        self.assertEqual(state["b"].ndim, 0)             # so state["b"][:] = v is an IndexError
        with self.assertRaises(IndexError):
            state["b"][:] = 1.0
        state["V"][:] = 99.0                             # writing a copy leaves the model alone
        self.assertFalse(np.allclose(model.V, 99.0))
        collapsed = " ".join(BASE_CANDIDATE_CONTRACT.split())
        self.assertIn("returns copies, not views", collapsed)
        self.assertIn("load_state_dict", collapsed)

    def test_contract_lists_the_real_candidate_output_fields(self):
        """Two iterations died on invented kwargs (valid_primary, train_trace)."""
        import dataclasses

        from src.agent.roles import BASE_CANDIDATE_CONTRACT
        from src.experiments.contracts import CandidateOutput

        collapsed = " ".join(BASE_CANDIDATE_CONTRACT.split())
        real = {f.name for f in dataclasses.fields(CandidateOutput)}
        for name in real:
            with self.subTest(field=name):
                self.assertIn(name, collapsed)
        # names the agents invented must not be real, and must be called out
        for invented in ("valid_primary", "train_trace"):
            with self.subTest(invented=invented):
                self.assertNotIn(invented, real)
                self.assertIn(invented, collapsed)

    def test_feature_families_are_in_required_coverage(self):
        # `multi_task` is registered and explorable but deliberately not a
        # convergence precondition: it has never produced a successful node, so
        # demanding it would make `converged` unreachable.
        self.assertEqual(
            coverage_families(),
            frozenset({"bpr", "group_softmax", "history_features"}),
        )
        self.assertIn("multi_task", FAMILIES)


if __name__ == "__main__":
    unittest.main()
