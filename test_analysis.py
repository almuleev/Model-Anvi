"""Regression checks for the bounded local code-analysis pipeline."""

import json
import queue
import runpy
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from code_analysis import CHUNK_CHARS, collect_code_chunks, split_code_chunk


CHAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHAT_DIR))
MODULE = runpy.run_path(str(CHAT_DIR / "local_chat.pyw"), run_name="test_import")
LocalChat = MODULE["LocalChat"]


class AnalysisTests(unittest.TestCase):
    def make_app(self):
        app = LocalChat.__new__(LocalChat)
        app.events = queue.Queue()
        app.response_lock = threading.Lock()
        app.active_connection = None
        app.active_response = None
        return app

    def test_modes_continue_without_growing_context(self):
        for key in ("off", "8x", "10x", "20x"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "sample.py").write_text("a = 1\n" * 10, encoding="utf-8")
                app = self.make_app()
                requests = []

                def fake_request(payload, request_id, cancel_event, stream_to_ui=False):
                    if not stream_to_ui:
                        return json.dumps({"summary": "Reviewed", "suspicions": []}), "stop"
                    requests.append(payload)
                    return ("First part.", "length") if len(requests) == 1 else ("End.", "stop")

                app.analysis_request_with_retries = fake_request
                app.run_code_analysis("Find errors", root, "sample.py", 1,
                                      threading.Event(), key)
                self.assertEqual(len(requests), 2)
                self.assertTrue(all(
                    request["options"]["num_ctx"] == 16384
                    and request["options"]["num_predict"] <= 5120
                    for request in requests
                ))
                self.assertFalse(requests[1]["think"])
                self.assertIn(("done", (1, None)), list(app.events.queue))

    def test_bad_json_on_one_line_is_reported_and_analysis_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("pass\n", encoding="utf-8")
            app = self.make_app()
            final_requests = []

            def fake_request(payload, request_id, cancel_event, stream_to_ui=False):
                if not stream_to_ui:
                    return "{broken JSON", "stop"
                final_requests.append(payload)
                return "Unresolved line noted.", "stop"

            app.analysis_request_with_retries = fake_request
            app.run_code_analysis("Find errors", root, "sample.py", 2,
                                  threading.Event(), "8x")
            self.assertIn("модель не вернула полный JSON",
                          final_requests[0]["messages"][0]["content"])
            self.assertIn(("done", (2, None)), list(app.events.queue))

    def test_exhausted_slices_get_a_closing_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("pass\n", encoding="utf-8")
            app = self.make_app()
            requests = []

            def fake_request(payload, request_id, cancel_event, stream_to_ui=False):
                if not stream_to_ui:
                    return json.dumps({"summary": "Reviewed", "suspicions": []}), "stop"
                requests.append(payload)
                if payload["options"]["num_predict"] == 1024:
                    return "Conclusion.", "stop"
                return "Still writing.", "length"

            app.analysis_request_with_retries = fake_request
            app.run_code_analysis("Find errors", root, "sample.py", 3,
                                  threading.Event(), "20x")
            self.assertEqual(len(requests), 9)
            self.assertEqual(requests[-1]["options"]["num_predict"], 1024)
            self.assertIn(("done", (3, None)), list(app.events.queue))
            self.assertFalse(any(
                kind == "token" and "незавершённым" in value[1]
                for kind, value in app.events.queue
            ))

    def test_exhausted_continuations_are_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("pass\n", encoding="utf-8")
            app = self.make_app()

            def fake_request(payload, request_id, cancel_event, stream_to_ui=False):
                if stream_to_ui:
                    return "Still writing.", "length"
                return json.dumps({"summary": "Reviewed", "suspicions": []}), "stop"

            app.analysis_request_with_retries = fake_request
            app.run_code_analysis("Review", root, "sample.py", 11,
                                  threading.Event(), "8x")
            events = list(app.events.queue)
            self.assertIn(("analysis_incomplete", (11, None)), events)
            self.assertNotIn(("done", (11, None)), events)

    def test_stream_interruption_returns_partial_for_continuation(self):
        app = self.make_app()

        class Response:
            status = 200

            def __iter__(self):
                yield json.dumps({"message": {"content": "Partial"}}).encode() + b"\n"

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

        with patch.dict(app.analysis_request.__globals__, {"HTTPConnection": Connection}):
            content, reason = app.analysis_request(
                {"stream": True, "messages": []}, 4, threading.Event(), True
            )
        self.assertEqual((content, reason), ("Partial", "interrupted"))

    def test_socket_request_does_not_hold_stop_lock(self):
        app = self.make_app()
        lock_available = []

        class Response:
            status = 200

            def __iter__(self):
                yield json.dumps({
                    "message": {"content": "OK"}, "done": True,
                    "done_reason": "stop",
                }).encode() + b"\n"

            def close(self):
                pass

        class Connection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                available = app.response_lock.acquire(blocking=False)
                lock_available.append(available)
                if available:
                    app.response_lock.release()

            def getresponse(self):
                return Response()

            def close(self):
                pass

        with patch.dict(app.analysis_request.__globals__, {"HTTPConnection": Connection}):
            result = app.analysis_request(
                {"stream": True, "messages": [{"role": "user", "content": "Hi"}]},
                10, threading.Event(), False,
            )
        self.assertEqual(result, ("OK", "stop"))
        self.assertEqual(lock_available, [True])

    def test_transient_backend_failure_is_retried(self):
        app = self.make_app()
        attempts = []

        def flaky(*args, **kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise OSError("temporary disconnect")
            return "Recovered", "stop"

        class Cancel:
            def is_set(self):
                return False

            def wait(self, seconds):
                return False

        app.analysis_request = flaky
        result = app.analysis_request_with_retries({"stream": True}, 5, Cancel())
        self.assertEqual(result, ("Recovered", "stop"))
        self.assertEqual(len(attempts), 3)

    def test_all_candidates_reach_review_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("pass\n" * 27, encoding="utf-8")
            chunks = [
                {"path": "sample.py", "start": start + 1, "end": start + 3,
                 "text": "".join(f"{line}: pass\n" for line in range(start + 1, start + 4))}
                for start in range(0, 27, 3)
            ]
            app = self.make_app()
            review_requests = []
            final_requests = []

            def fake_request(payload, request_id, cancel_event, stream_to_ui=False):
                if stream_to_ui:
                    final_requests.append(payload)
                    return "Complete.", "stop"
                if "format" not in payload:
                    review_requests.append(payload)
                    return "Candidate reviewed.", "stop"
                text = payload["messages"][0]["content"]
                import re
                lines = [int(n) for n in re.findall(r"(?m)^(\d+):", text)]
                return json.dumps({
                    "summary": "Reviewed",
                    "suspicions": [{"line": line, "reason": "check"} for line in lines],
                }), "stop"

            app.analysis_request_with_retries = fake_request
            with patch.dict(app.run_code_analysis.__globals__, {
                "collect_code_chunks": lambda *_: (chunks, [("sample.py", 27)])
            }):
                app.run_code_analysis("Find errors", root, "sample.py", 6,
                                      threading.Event(), "20x")
            self.assertEqual(len(review_requests), 7)
            self.assertIn("27 из 27", final_requests[0]["messages"][0]["content"])
            self.assertIn(("done", (6, None)), list(app.events.queue))

    def test_context_overflow_splits_source_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("pass\n" * 4, encoding="utf-8")
            app = self.make_app()
            source_requests = []

            def fake_request(payload, request_id, cancel_event, stream_to_ui=False):
                if stream_to_ui:
                    return "Finished.", "stop"
                source_requests.append(payload)
                if payload["messages"][0]["content"].count("pass") > 1:
                    raise RuntimeError("context length exceeded")
                return json.dumps({"summary": "One line reviewed", "suspicions": []}), "stop"

            app.analysis_request_with_retries = fake_request
            app.run_code_analysis("Review", root, "sample.py", 7,
                                  threading.Event(), "8x")
            self.assertEqual(len(source_requests), 7)
            self.assertTrue(all(p["options"]["num_ctx"] == 8192 for p in source_requests))
            self.assertIn(("done", (7, None)), list(app.events.queue))

    def test_long_single_line_can_be_split_without_losing_source(self):
        source = "x" * 2000
        parts = split_code_chunk({
            "path": "sample.py", "start": 1, "end": 1,
            "text": "1: " + source + "\n",
        })
        self.assertEqual(len(parts), 2)
        self.assertEqual("".join(part["text"][3:-1] for part in parts), source)
        self.assertTrue(all(part["start"] == part["end"] == 1 for part in parts))

    def test_long_source_line_is_segmented_during_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "x" * 20000
            (root / "sample.py").write_text(source + "\n", encoding="utf-8")
            chunks, stats = collect_code_chunks(root, "sample.py")
            self.assertEqual(stats, [("sample.py", 1)])
            self.assertEqual(len(chunks), 3)
            self.assertTrue(all(
                chunk["start"] == chunk["end"] == 1
                and len(chunk["text"]) <= CHUNK_CHARS
                for chunk in chunks
            ))
            self.assertEqual("".join(chunk["text"][3:-1] for chunk in chunks), source)
            parts = split_code_chunk({
                "path": "sample.py", "start": 1, "end": 1,
                "text": chunks[1]["text"] + chunks[2]["text"],
            })
            self.assertTrue(all(part["start"] == part["end"] == 1 for part in parts))

    def test_candidate_review_splits_on_context_overflow(self):
        app = self.make_app()
        requests = []

        def fake_request(payload, request_id, cancel_event, stream_to_ui=False):
            requests.append(payload)
            prompt = payload["messages"][0]["content"]
            if prompt.count("CANDIDATE") > 1:
                raise RuntimeError("context length exceeded")
            return "Reviewed one candidate.", "stop"

        app.analysis_request_with_retries = fake_request
        notes, issues = app.review_evidence_batch(
            "Review", [f"sample.py:{line} CANDIDATE\n{line}: pass" for line in range(1, 5)],
            8, threading.Event(),
        )
        self.assertEqual(len(notes), 4)
        self.assertFalse(issues)
        self.assertEqual(len(requests), 7)

    def test_long_candidate_line_is_not_falsely_confirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("x" * 2000 + "\n", encoding="utf-8")
            app = self.make_app()
            review_requests = []
            final_requests = []

            def fake_request(payload, request_id, cancel_event, stream_to_ui=False):
                if stream_to_ui:
                    final_requests.append(payload)
                    return "Needs manual review.", "stop"
                if "format" not in payload:
                    review_requests.append(payload)
                    return "Confirmed.", "stop"
                return json.dumps({
                    "summary": "Long source line",
                    "suspicions": [{"line": 1, "reason": "possible issue"}],
                }), "stop"

            app.analysis_request_with_retries = fake_request
            app.run_code_analysis("Find errors", root, "sample.py", 9,
                                  threading.Event(), "8x")
            self.assertFalse(review_requests)
            self.assertIn("слишком длинная для надёжной",
                          final_requests[0]["messages"][0]["content"])
            self.assertIn(("done", (9, None)), list(app.events.queue))


if __name__ == "__main__":
    unittest.main()
