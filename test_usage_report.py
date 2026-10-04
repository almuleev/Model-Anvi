"""Checks that Ollama's final counters reach the per-answer report."""

import json
import queue
import runpy
import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


CHAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHAT_DIR))
MODULE = runpy.run_path(str(CHAT_DIR / "local_chat.pyw"), run_name="test_import")
LocalChat = MODULE["LocalChat"]
UsageReport = MODULE["UsageReport"]


class UsageReportTests(unittest.TestCase):
    def make_app(self, mode="chat"):
        app = LocalChat.__new__(LocalChat)
        app.events = queue.Queue()
        app.response_lock = threading.Lock()
        app.active_connection = None
        app.active_response = None
        app.usage_report = UsageReport("2x", mode)
        return app

    def fake_connection(self, events):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()

            def __iter__(self):
                for event in events:
                    yield json.dumps(event).encode() + b"\n"

            def close(self):
                pass

        class Connection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return Response()

            def close(self):
                pass

        return Connection

    def test_chat_collects_exact_totals_and_phase_estimates(self):
        app = self.make_app()
        events = [
            {"message": {"thinking": "check"}},
            {"message": {"content": "answer"}},
            {"done": True, "done_reason": "stop", "prompt_eval_count": 42,
             "eval_count": 12, "load_duration": 1_000_000_000,
             "prompt_eval_duration": 2_000_000_000,
             "eval_duration": 3_000_000_000},
        ]
        with patch.dict(app.generate.__globals__, {"HTTPConnection": self.fake_connection(events)}):
            app.generate([{"role": "user", "content": "Hi"}], 1,
                         threading.Event(), None, "2x")
        row = app.usage_report.as_dict()["stages"]["chat"]
        self.assertEqual((row["requests"], row["input"], row["output"]), (1, 42, 12))
        self.assertEqual((row["load"], row["prompt"], row["generate"]), (1.0, 2.0, 3.0))
        self.assertEqual((row["thinking_chars"], row["answer_chars"]), (5, 6))
        self.assertIn(("done", (1, None)), list(app.events.queue))

    def test_interrupted_analysis_request_does_not_invent_token_count(self):
        app = self.make_app("analysis")
        app.usage_report.switch("scan")
        events = [{"message": {"content": "Partial"}}]
        with patch.dict(app.analysis_request.__globals__,
                        {"HTTPConnection": self.fake_connection(events)}):
            content, reason = app.analysis_request(
                {"stream": True, "messages": []}, 2, threading.Event())
        self.assertEqual((content, reason), ("Partial", "interrupted"))
        row = app.usage_report.as_dict()["stages"]["scan"]
        self.assertEqual((row["requests"], row["missing"], row["output"]), (1, 1, 0))

    def test_usage_report_table_is_created_without_changing_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            @contextmanager
            def managed_connection():
                db = sqlite3.connect(database)
                try:
                    with db:
                        yield db
                finally:
                    db.close()

            with patch.dict(MODULE["init_db"].__globals__, {"db_connect": managed_connection}):
                MODULE["init_db"]()
                with managed_connection() as db:
                    tables = {row[0] for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'")}
                    self.assertTrue({"messages", "usage_reports"}.issubset(tables))


if __name__ == "__main__":
    unittest.main()
