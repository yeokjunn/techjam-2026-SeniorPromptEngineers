"""The atomic writes must survive the dashboard reading the same file.

`docs/UI_QUICKSTART.md` tells the operator to run the read-only dashboard in one terminal and
the agent in another. On Windows that is a collision by construction: `os.replace` raises
PermissionError (WinError 5) while another process holds the destination open, because Python's
`open()` does not request FILE_SHARE_DELETE. An observed production run lost 2 of its 7
iterations to exactly this, each recorded as `harness_error`.

No unit test covered it because it only appears with two processes (or threads) live at once.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from src.agent.audit import ResearchAudit, _replace_atomic


class ConcurrentReaderTests(unittest.TestCase):
    WRITES = 60
    POLL_SECONDS = 0.01  # the real dashboard polls every 5 s; this is 500x more aggressive

    def _run_under_reader(self, write, path: Path) -> tuple[int, int]:
        stop = threading.Event()

        def poller() -> None:
            while not stop.is_set():
                try:
                    # Closed promptly, like the dashboard's own reader: the collision window
                    # is the replace itself, not a leaked handle.
                    with path.open("r", encoding="utf-8") as handle:
                        handle.read()
                except OSError:
                    pass
                time.sleep(self.POLL_SECONDS)

        reader = threading.Thread(target=poller, daemon=True)
        reader.start()
        succeeded = failed = 0
        try:
            for index in range(self.WRITES):
                try:
                    write(path, {"stage": f"iteration-{index}"})
                    succeeded += 1
                except PermissionError:
                    failed += 1
                time.sleep(0.002)
        finally:
            stop.set()
            reader.join(timeout=2)
        return succeeded, failed

    def test_json_writes_survive_a_concurrent_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "activity.json"
            ResearchAudit.write_json_atomic(target, {"stage": "init"})
            succeeded, failed = self._run_under_reader(
                ResearchAudit.write_json_atomic, target
            )
        self.assertEqual(failed, 0, "atomic JSON write lost a race with a reader")
        self.assertEqual(succeeded, self.WRITES)

    def test_text_writes_survive_a_concurrent_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "journal.md"
            ResearchAudit.write_text_atomic(target, "init")
            succeeded, failed = self._run_under_reader(
                lambda path, value: ResearchAudit.write_text_atomic(path, str(value)), target
            )
        self.assertEqual(failed, 0, "atomic text write lost a race with a reader")
        self.assertEqual(succeeded, self.WRITES)

    def test_a_genuine_permission_error_still_propagates(self):
        """The retry must not swallow a fault that will never clear."""
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "no_such_dir" / "activity.json"
            with self.assertRaises((PermissionError, FileNotFoundError, OSError)):
                _replace_atomic(Path(directory) / "absent.tmp", missing)

    def test_the_written_value_is_intact_after_contention(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "activity.json"
            ResearchAudit.write_json_atomic(target, {"stage": "init"})
            self._run_under_reader(ResearchAudit.write_json_atomic, target)
            import json

            restored = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(restored["stage"], f"iteration-{self.WRITES - 1}")


if __name__ == "__main__":
    unittest.main()
