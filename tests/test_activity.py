from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.activity import summarize_role_output
from src.agent.audit import ResearchAudit


class ActivityAuditTests(unittest.TestCase):
    def test_activity_snapshot_is_atomic_and_timeline_has_transition_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            handle = audit.start_activity(
                3,
                "builder",
                role="builder",
                experiment_id="candidate_bpr",
                objective="Build the approved candidate.",
            )
            active = json.loads((audit.run_dir / "activity.json").read_text(encoding="utf-8"))
            self.assertEqual(active["status"], "active")
            audit.finish_activity(
                handle,
                agent_note={"decision": "Candidate generated."},
            )
            current = json.loads((audit.run_dir / "activity.json").read_text(encoding="utf-8"))
            timeline = [
                json.loads(line)
                for line in (audit.run_dir / "activity.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(current["status"], "completed")
            self.assertEqual([item["status"] for item in timeline], ["active", "completed"])
            self.assertEqual({item["event_id"] for item in timeline}, {handle.event_id})

    def test_change_capture_writes_summary_and_reproducible_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = ResearchAudit(Path(directory) / "run")
            summary = audit.record_candidate_changes(
                1,
                "candidate_bpr",
                {"candidate.py": "x = 2\ny = 3\n", "test_candidate.py": "assert True\n"},
                {"candidate.py": "x = 1\n"},
            )
            patch = (audit.run_dir / summary["patch_path"]).read_text(encoding="utf-8")
            self.assertIn("-x = 1", patch)
            self.assertIn("+x = 2", patch)
            self.assertEqual(summary["lines_added"], 3)
            self.assertEqual(summary["lines_deleted"], 1)
            self.assertTrue((audit.run_dir / "changes" / "001_candidate_bpr.json").is_file())

    def test_agent_notes_are_allowlisted_and_redacted(self):
        note = summarize_role_output(
            "researcher",
            {
                "hypothesis": "Try BPR.",
                "rationale": "api_key=do-not-log",
                "private_chain_of_thought": "must not be exposed",
                "evidence": [{"title": "Paper", "url": "https://example.test"}],
            },
        )
        self.assertNotIn("private_chain_of_thought", note)
        self.assertNotIn("do-not-log", json.dumps(note))
        self.assertEqual(note["evidence"][0]["title"], "Paper")


if __name__ == "__main__":
    unittest.main()
