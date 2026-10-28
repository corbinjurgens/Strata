"""Model discovery, on-demand loading and safe unloading over the actual HTTP API."""
import json
import io
import os
import subprocess
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve import server
from serve.server import ByteTokenizer, EngineDied, EngineStuck, MockEngine, Service, StrataEngine, serve


class ResidentEngine(StrataEngine):
    def __init__(self, tok):
        super().__init__("fixture", [], lazy=True)
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.loaded, self.starts, self.closes = False, 0, 0
        self.unloaded = True
        self.progress = None

    def alive(self):
        return self.loaded

    def exit_code(self):
        return None

    def restart(self):
        self.loaded = True
        self.starts += 1
        self.unloaded = False

    def close(self):
        self.loaded = False
        self.closes += 1

    def unload(self):
        self.close()
        self.unloaded = True

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


class Lifecycle(unittest.TestCase):
    def test_close_cleans_up_after_a_broken_stdin_pipe(self):
        for running in (False, True):
            with self.subTest(running=running):
                engine = StrataEngine("missing-executable", [], lazy=True)
                read_fd, write_fd = os.pipe()
                stdin = os.fdopen(write_fd, "w")
                self.addCleanup(stdin.close)
                os.close(read_fd)                       # the engine's end of the pipe is already gone
                stdin.write("pending")                  # buffered: flush() and close() will hit the broken pipe
                proc = mock.Mock(stdin=stdin, stdout=io.StringIO())
                self.addCleanup(proc.stdout.close)
                proc.poll.side_effect = [None if running else 0, 0]
                engine.proc, engine.pump, engine.log = proc, mock.Mock(), io.StringIO()
                engine.pump.is_alive.return_value = False
                self.addCleanup(engine.log.close)
                engine.ended, engine.progress, engine.last = False, (1, 2), {"generated": 1}

                engine.close()

                self.assertTrue(stdin.closed)
                self.assertTrue(proc.stdout.closed)
                self.assertTrue(engine.log.closed)
                engine.pump.join.assert_called_once_with(timeout=2)
                self.assertIsNone(engine.proc)
                self.assertTrue(engine.ended)
                self.assertIsNone(engine.progress)
                self.assertEqual(engine.last, {})
                if running:
                    proc.terminate.assert_called_once()
                    proc.wait.assert_called_once_with(timeout=20)
                else:
                    proc.terminate.assert_not_called()
                    proc.wait.assert_not_called()
                proc.kill.assert_not_called()
                engine.close()                          # repeated cleanup is harmless

    def test_close_sends_eof_before_waiting_for_windows_reader(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = lambda **kwargs: self.assertTrue(proc.stdin.closed)
        engine.proc = proc
        engine.close()
        self.assertIsNone(engine.proc)
        proc.kill.assert_not_called()

    def test_close_does_not_forget_a_process_still_exiting(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("engine", 2)
        engine.proc = proc
        with self.assertRaisesRegex(EngineStuck, "still releasing"):
            engine.close()
        self.assertIs(engine.proc, proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_close_terminates_before_it_kills(self):
        # an engine that does not end on QUIT is terminated first (as the unload always did); kill is the last resort
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = [subprocess.TimeoutExpired("engine", 20), 0]
        engine.proc = proc
        engine.close()
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        self.assertIsNone(engine.proc)

    def test_native_lazy_constructor_does_not_start_a_process(self):
        engine = StrataEngine("missing-executable", ["--max-context", "16384", "--batch", "4"], lazy=True)
        self.assertFalse(engine.alive())
        self.assertEqual(engine.max_context, 16384)
        self.assertIsNone(engine.exit_code())
        self.assertEqual((engine.requested_slots, engine.batch), (4, 0))
        engine.close()

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def test_discover_chat_unload_and_stream_reload(self):
        self.assertEqual(self.request("/api/health")[1]["service"], "strata")
        self.assertFalse(self.request("/v1/status")[1]["loaded"])
        self.assertEqual(self.request("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertTrue(self.request("/props")[1]["models_autoload"])
        self.assertEqual(self.engine.starts, 0)
        chat = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "reasoning_effort": "none"}
        for _ in range(2):
            code, answer = self.request("/v1/chat/completions", chat)
            self.assertEqual(code, 200)
            self.assertEqual(answer["choices"][0]["message"]["content"], "Hello.")
        self.assertEqual(self.engine.starts, 1)
        self.assertFalse(self.request("/v1/unload", {})[1]["loaded"])
        code, stream = self.request("/v1/chat/completions", {**chat, "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("data: [DONE]", stream)
        self.assertEqual(self.engine.starts, 2)

    def test_busy_auth_and_foreign_origin_cannot_unload(self):
        self.engine.loaded = True
        with self.svc.fifo:
            self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.svc.api_key = "local-secret"
        self.assertEqual(self.request("/v1/unload", {})[0], 401)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret",
                                                        "Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.engine.closes, 0)
        self.assertTrue(self.engine.loaded)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret"})[0], 200)

    def test_an_engine_that_died_on_load_is_reported_as_such(self):
        # EngineDied is a RuntimeError: it keeps its own answer ("the next request restarts it")
        def died():
            raise EngineDied("the engine ended while it started")
        self.engine.restart = died
        code, body = self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(code, 503)
        self.assertIn("the next request restarts it", body["error"]["message"])

    def test_explicit_load_and_model_matching(self):
        self.assertEqual(self.request("/v1/load", {"model": "other-model"})[0], 404)
        self.assertEqual(self.engine.starts, 0)
        before = time.time()
        code, status = self.request("/v1/load", {"model": self.svc.model})
        self.assertEqual(code, 200)
        self.assertTrue(status["loaded"])
        self.assertEqual(self.engine.starts, 1)
        self.assertGreaterEqual(self.svc.last_request_at, before)
        self.svc.status["queued"] = 1
        self.assertEqual(self.request("/v1/load", {})[0], 409)
        self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.assertEqual(self.engine.closes, 0)
        self.assertEqual(self.request("/v1/load", {}, {"Content-Type": "text/plain"})[0], 415)

    def test_loading_is_shared_by_lazy_chats_and_controls_are_busy(self):
        entered, release = threading.Event(), threading.Event()
        restart = self.engine.restart
        self.svc.set_aliases(["alias"])
        def slow_load():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("load gate timed out")
            restart()
        self.engine.restart = slow_load
        results = []
        chat = {"messages": [{"role": "user", "content": "Hello"}], "reasoning_effort": "none", "max_tokens": 16}
        threads = [threading.Thread(target=lambda: results.append(self.request("/v1/chat/completions", chat)))
                   for _ in range(2)]
        try:
            threads[0].start()
            self.assertTrue(entered.wait(3))
            threads[1].start()
            status = self.request("/v1/status")[1]
            self.assertEqual((status["lifecycle"], status["loaded"]), ("loading", False))
            models = self.request("/v1/models")[1]["data"]
            self.assertEqual([m["id"] for m in models], [self.svc.model, "alias"])
            self.assertEqual([m["status"]["value"] for m in models], ["loading", "loading"])
            for path in ("/load", "/unload", "/v1/load", "/v1/unload"):
                code, body = self.request(path, {})
                self.assertEqual((code, body["error"]["code"]), (409, "busy"), path)
            self.assertEqual(results, [], "first chats must wait until loading completes")
        finally:
            release.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual([code for code, _ in results], [200, 200])
        self.assertEqual(self.engine.starts, 1)
        self.assertEqual(self.request("/v1/status")[1]["lifecycle"], "loaded")

    def test_unloading_is_synchronous_and_death_stays_listed_without_idle(self):
        self.engine.restart()
        self.svc.set_aliases(["alias"])
        entered, release = threading.Event(), threading.Event()
        unload = self.engine.unload
        def slow_unload():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("unload gate timed out")
            unload()
        self.engine.unload = slow_unload
        result = []
        thread = threading.Thread(target=lambda: result.append(self.request("/v1/unload", {"model": "alias"})))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(result, [])
            status = self.request("/v1/status")[1]
            self.assertEqual((status["lifecycle"], status["loaded"]), ("unloading", False))
            self.assertEqual([m["status"]["value"] for m in self.request("/v1/models")[1]["data"]],
                             ["unloaded", "unloaded"])
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual((result[0][0], result[0][1]["status"], result[0][1]["lifecycle"]),
                         (200, "unloaded", "unloaded"))
        self.engine.restart()
        self.engine.loaded = False                         # unexpected child death, no idle unload configured
        self.assertEqual(self.request("/v1/status")[1]["lifecycle"], "failed")
        self.assertEqual([m["id"] for m in self.request("/v1/models")[1]["data"]], [self.svc.model, "alias"])
        self.assertEqual(self.request("/models")[1]["data"][0]["status"]["value"], "unloaded")

    def test_runtime_codes_match_json_and_started_streams(self):
        for failure, code in ((EngineDied("engine went away"), "engine_died"),
                              (server.NativeFailure("out_of_memory"), "out_of_memory"),
                              (server.NativeFailure("insufficient_resources"), "insufficient_resources")):
            def generate(*args, **kwargs):
                yield 88
                raise failure
            self.engine.generate = generate
            for api in ("chat/completions", "messages", "responses"):
                for stream in (False, True):
                    with self.subTest(code=code, api=api, stream=stream):
                        body = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32,
                                "reasoning_effort": "none", "stream": stream}
                        if api == "responses":
                            body = {"input": "Hello", "max_output_tokens": 32, "stream": stream,
                                    "reasoning": {"effort": "none"}}
                        status, answer = self.request("/v1/" + api, body)
                        if stream:
                            self.assertEqual(status, 200)
                            events = [json.loads(line[6:]) for line in answer.splitlines()
                                      if line.startswith("data: {")]
                            errors = [event.get("error") or event.get("response", {}).get("error")
                                      for event in events]
                            self.assertIn(code, [error["code"] for error in errors if error])
                        else:
                            self.assertEqual((status, answer["error"]["code"]), (503, code))

    def test_restart_failure_is_json_on_both_control_routes(self):
        def refuse():
            raise RuntimeError("startup exited without READY")
        self.engine.restart = refuse
        for path in ("/load", "/v1/load", "/v1/chat/completions"):
            code, body = self.request(path, {"messages": [{"role": "user", "content": "Hi"}]})
            self.assertEqual((code, body["error"]["code"]), (503, "engine_died"))
            self.assertEqual(self.request("/v1/status")[1]["lifecycle"], "failed")

    def test_queued_and_still_cancelling_requests_reject_all_controls(self):
        self.engine.restart()
        for state in ("queued", "slot"):
            if state == "queued":
                self.svc.status["queued"] = 1
            else:
                self.svc.status["queued"] = 0
                self.engine.slot_busy = [True]
            for path in ("/load", "/unload", "/v1/load", "/v1/unload"):
                code, body = self.request(path, {})
                self.assertEqual((code, body["error"]["code"]), (409, "busy"), (state, path))
        self.assertEqual(self.engine.closes, 0)

    def test_uncertain_exit_is_failed_and_refuses_restart_until_unloaded(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        self.svc.engine = engine
        proc = mock.Mock(stdin=io.StringIO(), stdout=io.StringIO())
        self.addCleanup(proc.stdin.close)
        self.addCleanup(proc.stdout.close)
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("engine", 1)
        engine.proc = proc
        engine.ended, engine.unloaded = False, False
        code, body = self.request("/v1/unload", {})
        self.assertEqual((code, body["error"]["code"]), (503, "engine_died"))
        self.assertIs(engine.proc, proc)
        self.assertFalse(engine.alive())
        self.assertEqual(self.request("/v1/status")[1]["lifecycle"], "failed")
        self.assertEqual(self.request("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        with mock.patch.object(server, "popen", side_effect=AssertionError("replacement spawned")):
            self.assertEqual(self.request("/v1/load", {})[1]["error"]["code"], "engine_died")
            with self.assertRaises(EngineStuck):
                engine.restart()
        proc.poll.return_value = 0
        code, body = self.request("/v1/unload", {})
        self.assertEqual((code, body["status"], body["lifecycle"]), (200, "unloaded", "unloaded"))
        self.assertIsNone(engine.proc)


STARTUP = r'''import sys
mode = sys.argv[sys.argv.index('--mode') + 1]
if mode != 'generic':
    print('ERROR ' + mode + ': refused fixture allocation', file=sys.stderr, flush=True)
else:
    print('arbitrary stderr says out of memory; no typed evidence', file=sys.stderr, flush=True)
'''


class NativeStartup(unittest.TestCase):
    def test_typed_startup_records_and_generic_exit_before_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake.py"
            script.write_text(STARTUP, encoding="utf-8")
            real_popen = server.popen
            children = []
            def spawn(what, argv, **kwargs):
                child = real_popen(what, [sys.executable, "-B", str(script), *argv[1:]], **kwargs)
                children.append(child)
                return child
            cases = (("out_of_memory", "out_of_memory"), ("insufficient_resources", "insufficient_resources"),
                     ("invalid_launch_plan", "invalid_launch_plan"), ("generic", "engine_died"))
            for mode, expected, logging in ((mode, expected, logging) for logging in (False, True) for mode, expected in cases):
                with self.subTest(mode=mode, logging=logging), mock.patch.object(server, "popen", side_effect=spawn), \
                        mock.patch.object(StrataEngine, "RESTART_RETRY_S", 0):
                    engine = StrataEngine("fixture", ["--mode", mode],
                                          log=str(Path(directory) / (mode + ".log")) if logging else None, lazy=True)
                    tok = ByteTokenizer()
                    svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
                    http = serve(svc, port=0)
                    req = urllib.request.Request(f"http://127.0.0.1:{http.server_address[1]}/v1/load", data=b"{}",
                                                 headers={"Content-Type": "application/json"})
                    try:
                        with self.assertRaises(urllib.error.HTTPError) as caught:
                            urllib.request.urlopen(req, timeout=10)
                        with caught.exception as response:
                            self.assertEqual((response.code, json.load(response)["error"]["code"]), (503, expected))
                        self.assertEqual(svc.v1_status()["lifecycle"], "failed")
                        self.assertIsNone(engine.proc)
                        self.assertTrue(all(child.poll() is not None and child.stdin.closed and child.stdout.closed
                                            for child in children))
                    finally:
                        http.shutdown()
                        http.server_close()
                        engine.close()


class ReaderCleanup(unittest.TestCase):
    def test_stdin_cleanup_retains_blocked_flush_and_close_until_retry(self):
        for operation in ('quit', 'close'):
            for exited in (False, True):
                with self.subTest(operation=operation, exited=exited):
                    class Sink(io.RawIOBase):
                        def __init__(self):
                            super().__init__()
                            self.entered, self.release = threading.Event(), threading.Event()
                        def writable(self): return True
                        def write(self, data):
                            self.entered.set()
                            if not self.release.wait(12): raise AssertionError('fixture input not released')
                            return len(data)
                    sink = Sink()
                    stdin = io.TextIOWrapper(io.BufferedWriter(sink), encoding='utf-8')
                    if operation == 'close': stdin.write('pending')
                    class Process:
                        def __init__(self):
                            self.stdin, self.stdout, self.stderr = stdin, io.StringIO(), None
                            self.rc = 0 if exited else None
                        def poll(self): return self.rc
                        def wait(self, timeout):
                            if self.rc is None: raise subprocess.TimeoutExpired('fixture', timeout)
                            return self.rc
                        def terminate(self): pass
                        def kill(self): pass
                    proc = Process()
                    engine = StrataEngine('fixture', [], lazy=True)
                    engine.proc, engine.unloaded = proc, False
                    svc = Service(engine, ByteTokenizer(), None)
                    def post(path):
                        handler = object.__new__(server.make_handler(svc))
                        handler.path = path
                        handler._authorized = lambda: True
                        handler._foreign_page = lambda: False
                        handler._own_page = lambda *args: True
                        handler._body = lambda: b'{}'
                        result = []
                        handler._json = lambda code, body: result.append((code, body))
                        handler._post()
                        return result[0]
                    result, done = [], threading.Event()
                    close = StrataEngine.close
                    def unload():
                        try: result.append(post('/v1/unload'))
                        finally: done.set()
                    with mock.patch.object(StrataEngine, 'close', lambda self: close(self, timeout=.01)):
                        caller = threading.Thread(target=unload, daemon=True)
                        caller.start()
                        try:
                            # Already-exited input without pending bytes does not write.
                            if exited and operation == 'quit':
                                self.assertTrue(done.wait(3))
                                self.assertEqual(result[0][0], 200)
                            else:
                                self.assertTrue(sink.entered.wait(1))
                                self.assertTrue(done.wait(3), 'cleanup blocked on stdin')
                                self.assertEqual((result[0][0], result[0][1]['error']['code']), (503, 'engine_died'))
                                self.assertIs(engine.proc, proc)
                                self.assertTrue(engine.termination_uncertain)
                                self.assertEqual(svc.lifecycle(), 'failed')
                                self.assertFalse(svc.fifo.locked())
                                with mock.patch.object(server, 'popen', side_effect=AssertionError('replacement')):
                                    self.assertEqual(post('/v1/load')[0], 503)
                        finally:
                            proc.rc = 0
                            sink.release.set()
                            caller.join(3)
                            close(engine, timeout=.01)
                    self.assertFalse(caller.is_alive())
                    self.assertTrue(stdin.closed and proc.stdout.closed)
                    self.assertIsNone(engine.proc)

    def test_planned_load_timeout_retains_active_stdin_writer_before_cleanup_retry(self):
        import base64
        from serve import launch_plan, launch_runtime
        from serve.test_launch_plan import fake_metadata
        metadata = fake_metadata(['fixture', '--resolve-launch', '--max-context', '8192', '--prefill', '64',
                                  '--resident-budget-gib', '1', '--plan-vram-free-bytes', '1000000',
                                  '--plan-ram-available-bytes', '1000000', '--plan-commit-available-bytes', '1000000'], {})
        plan = {'plan_digest': '0' * 64, 'requested': {}, 'estimates': {},
                'launch': {'executable': 'fixture', 'argv': ['fixture', '--serve', '--native', 'fixture.gguf'],
                           'cwd': None, 'environment': {}},
                'native': {'native_payload_base64': base64.b64encode(metadata['native_payload'].encode()).decode()}}
        # A tiny buffer blocks in write; the larger buffer blocks in flush.
        for held, buffer_size, exits in ((False, 8192, True), (True, 1, True),
                                         (True, 8192, True), (True, 8192, False)):
            with self.subTest(held=held, buffer_size=buffer_size, exits=exits):
                class HeldWrite(io.RawIOBase):
                    def __init__(self):
                        super().__init__()
                        self.entered, self.release = threading.Event(), threading.Event()
                    def writable(self): return True
                    def write(self, data):
                        self.entered.set()
                        if not self.release.wait(15): raise AssertionError('fixture write not released')
                        return len(data)
                class Input(io.TextIOWrapper):
                    def __init__(self, raw):
                        super().__init__(io.BufferedWriter(raw, buffer_size), encoding='utf-8')
                        self.close_entered = threading.Event()
                    def close(self):
                        self.close_entered.set()
                        return super().close()
                raw = HeldWrite()
                if not held: raw.release.set()
                class Process:
                    def __init__(self):
                        self.stdin, self.stdout, self.stderr, self.rc = Input(raw), io.StringIO(), None, None
                    def poll(self): return self.rc
                    def wait(self, timeout):
                        if not exits: raise subprocess.TimeoutExpired('fixture', timeout)
                        self.rc = 0
                        return 0
                    def terminate(self):
                        if exits: self.rc = -1
                    def kill(self):
                        if exits: self.rc = -9
                proc = Process()
                with mock.patch.object(launch_plan, 'validate_plan'), \
                        mock.patch.object(launch_runtime, 'admit', return_value=None), \
                        mock.patch.object(launch_plan, 'replay_environment', return_value={}), \
                        mock.patch.object(server, 'popen', return_value=proc) as spawn, \
                        mock.patch.object(server, 'contain'), mock.patch.object(server, 'ENGINE_READY_S', .02):
                    engine = StrataEngine('fixture', [], lazy=True, launch_plan=plan)
                    svc = Service(engine, ByteTokenizer(), None)
                    def post(path):
                        handler = object.__new__(server.make_handler(svc))
                        handler.path = path
                        handler._authorized = lambda: True
                        handler._foreign_page = lambda: False
                        handler._own_page = lambda *args: True
                        handler._body = lambda: b'{}'
                        result = []
                        handler._json = lambda code, body: result.append((code, body))
                        handler._post()
                        return result[0]
                    result, done = [], threading.Event()
                    def load():
                        try: result.append(post('/v1/load'))
                        finally: done.set()
                    caller = threading.Thread(target=load, daemon=True)
                    caller.start()
                    try:
                        self.assertTrue(raw.entered.wait(1))
                        self.assertTrue(done.wait(3), 'load blocked on the strict-frame writer')
                        self.assertEqual((result[0][0], result[0][1]['error']['code']), (503, 'engine_died'))
                        self.assertEqual(svc.lifecycle(), 'failed')
                        self.assertFalse(svc.fifo.locked())
                        if held:
                            self.assertIs(engine.proc, proc)
                            self.assertTrue(engine.termination_uncertain)
                            self.assertTrue(engine.ready_reader.is_alive())
                            self.assertFalse(proc.stdin.close_entered.is_set())
                            self.assertFalse(proc.stdin.closed or proc.stdout.closed)
                            self.assertEqual(post('/v1/load')[0], 503)
                            self.assertEqual(spawn.call_count, 1)
                        else:
                            self.assertIsNone(engine.proc)
                    finally:
                        raw.release.set()
                        caller.join(3)
                        engine.ready_reader.join(2)
                        proc.rc = 0
                        engine.close(timeout=.01)
                    self.assertFalse(caller.is_alive())
                    self.assertFalse(engine.ready_reader.is_alive())
                    self.assertTrue(proc.stdin.closed and proc.stdout.closed)
                    self.assertIsNone(engine.proc)
                    self.assertFalse(engine.termination_uncertain)

    def test_public_load_timeout_retains_reader_and_releases_fifo_before_retry(self):
        from serve import launch_runtime
        for held in (False, True):
            with self.subTest(held=held):
                class HeldRead(io.RawIOBase):
                    def __init__(self):
                        super().__init__()
                        self.entered, self.release = threading.Event(), threading.Event()
                    def readable(self): return True
                    def readinto(self, buffer):
                        self.entered.set()
                        if not self.release.wait(10): raise AssertionError('fixture startup not released')
                        return 0
                raw = HeldRead()
                if not held: raw.release.set()
                class Process:
                    def __init__(self):
                        self.stdin, self.stdout, self.stderr = io.StringIO(), io.TextIOWrapper(io.BufferedReader(raw), encoding='utf-8'), None
                        self.rc = None
                    def poll(self): return self.rc
                    def wait(self, timeout): self.rc = 0; return 0
                    def terminate(self): self.rc = -1
                    def kill(self): self.rc = -9
                proc = Process()
                engine = StrataEngine('fixture', [], lazy=True)
                svc = Service(engine, ByteTokenizer(), None)
                def post(path):
                    handler = object.__new__(server.make_handler(svc))
                    handler.path = path
                    handler._authorized = lambda: True
                    handler._foreign_page = lambda: False
                    handler._own_page = lambda *args: True
                    handler._body = lambda: b'{}'
                    answer = []
                    handler._json = lambda code, body: answer.append((code, body))
                    handler._post()
                    return answer[0]
                result, done = [], threading.Event()
                restart = StrataEngine.restart
                def load():
                    try: result.append(post('/v1/load'))
                    finally: done.set()
                with mock.patch.object(server, 'popen', return_value=proc), mock.patch.object(server, 'contain'), \
                        mock.patch.object(launch_runtime, 'build_record', return_value=None), \
                        mock.patch.object(server, 'ENGINE_READY_S', .02), \
                        mock.patch.object(StrataEngine, 'restart', lambda self: restart(self, tries=1)):
                    caller = threading.Thread(target=load, daemon=True)
                    caller.start()
                    try:
                        self.assertTrue(raw.entered.wait(1))
                        self.assertTrue(done.wait(3), 'load timeout blocked on readiness cleanup')
                        self.assertEqual((result[0][0], result[0][1]['error']['code']), (503, 'engine_died'))
                        self.assertEqual(svc.lifecycle(), 'failed')
                        self.assertFalse(svc.fifo.locked())
                        if held:
                            self.assertIs(engine.proc, proc)
                            self.assertTrue(engine.termination_uncertain)
                            self.assertTrue(engine.ready_reader.is_alive())
                            self.assertFalse(proc.stdout.closed)
                            with mock.patch.object(server, 'popen', side_effect=AssertionError('replacement')):
                                self.assertEqual(post('/v1/load')[0], 503)
                        else:
                            self.assertIsNone(engine.proc)
                    finally:
                        raw.release.set()
                        caller.join(3)
                        engine.ready_reader.join(2)
                        engine.close(timeout=.01)
                self.assertFalse(caller.is_alive())
                self.assertFalse(engine.ready_reader.is_alive())
                self.assertTrue(proc.stdin.closed and proc.stdout.closed)
                self.assertIsNone(engine.proc)

    def test_public_unload_retains_active_stdout_readers_and_retries_after_eof(self):
        import queue
        for reader_kind in ('pump', 'ready_reader'):
            for held in (False, True):
                with self.subTest(reader=reader_kind, held=held):
                    class HeldRead(io.RawIOBase):
                        def __init__(self):
                            super().__init__()
                            self.entered, self.release = threading.Event(), threading.Event()
                        def readable(self): return True
                        def readinto(self, buffer):
                            self.entered.set()
                            if not self.release.wait(8): raise AssertionError('fixture stdout not released')
                            return 0
                    raw = HeldRead()
                    if not held: raw.release.set()
                    stdout = io.TextIOWrapper(io.BufferedReader(raw), encoding='utf-8')
                    proc = mock.Mock(stdin=io.StringIO(), stdout=stdout, stderr=None)
                    proc.poll.return_value = 0
                    engine = StrataEngine('fixture', [], lazy=True)
                    engine.proc, engine.lines, engine.slot_q, engine.gen, engine.unloaded = proc, queue.Queue(), [], 1, False
                    target = engine._pump if reader_kind == 'pump' else lambda: engine._ready_pump(proc, queue.Queue())
                    reader = threading.Thread(target=target, daemon=True)
                    setattr(engine, reader_kind, reader)
                    reader.start()
                    self.assertTrue(raw.entered.wait(1))
                    svc = Service(engine, ByteTokenizer(), None)
                    def post(path):
                        handler = object.__new__(server.make_handler(svc))
                        handler.path = path
                        handler._authorized = lambda: True
                        handler._foreign_page = lambda: False
                        handler._own_page = lambda *args: True
                        handler._body = lambda: b'{}'
                        answer = []
                        handler._json = lambda code, body: answer.append((code, body))
                        handler._post()
                        return answer[0]
                    result, done = [], threading.Event()
                    def unload():
                        try: result.append(post('/v1/unload'))
                        finally: done.set()
                    closer = threading.Thread(target=unload, daemon=True)
                    closer.start()
                    try:
                        self.assertTrue(done.wait(3), 'public unload blocked on the stdout reader')
                        code, body = result[0]
                        if held:
                            self.assertEqual((code, body['error']['code']), (503, 'engine_died'))
                            self.assertIs(engine.proc, proc)
                            self.assertIs(getattr(engine, reader_kind), reader)
                            self.assertTrue(engine.termination_uncertain)
                            self.assertEqual(svc.lifecycle(), 'failed')
                            self.assertFalse(stdout.closed)
                            with mock.patch.object(server, 'popen', side_effect=AssertionError('replacement')):
                                self.assertEqual(post('/v1/load')[0], 503)
                        else:
                            self.assertEqual((code, body['status']), (200, 'unloaded'))
                            self.assertIsNone(engine.proc)
                    finally:
                        raw.release.set()
                        closer.join(2)
                        reader.join(2)
                        try:
                            self.assertFalse(reader.is_alive())
                            self.assertTrue(stdout.closed, 'stdout reader must close its stream before retry')
                        finally:
                            engine.close(timeout=.01)
                    self.assertFalse(closer.is_alive())
                    self.assertIsNone(engine.proc)
                    self.assertFalse(engine.termination_uncertain)

    def test_live_stderr_reader_retains_cleanup_ownership_without_blocking_close(self):
        class HeldRead(io.RawIOBase):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()
            def readable(self): return True
            def readinto(self, buffer):
                self.entered.set()
                if not self.release.wait(8): raise AssertionError('fixture read not released')
                return 0
        raw = HeldRead()
        stderr = io.TextIOWrapper(io.BufferedReader(raw), encoding='utf-8')
        proc = mock.Mock(stdin=io.StringIO(), stdout=io.StringIO(), stderr=stderr)
        proc.poll.return_value = 0
        engine = StrataEngine('fixture', [], lazy=True)
        engine.proc = proc
        engine.stderr_reader = threading.Thread(target=engine._stderr_pump, args=(proc,), daemon=True)
        engine.stderr_reader.start()
        self.assertTrue(raw.entered.wait(1))
        result, complete = [], threading.Event()
        def close():
            try:
                engine.close(timeout=.01)
                result.append('returned')
            except Exception as error:
                result.append(type(error))
            finally:
                complete.set()
        closer = threading.Thread(target=close, daemon=True)
        closer.start()
        try:
            self.assertTrue(complete.wait(3), 'close blocked on a live reader after its bounded join')
            self.assertEqual(result, [EngineStuck])
            self.assertIs(engine.proc, proc)
            self.assertTrue(engine.termination_uncertain)
            self.assertFalse(stderr.closed)
            with self.assertRaises(EngineStuck), mock.patch.object(server, 'popen', side_effect=AssertionError('replacement')):
                engine.restart()
        finally:
            raw.release.set()
            closer.join(2)
            engine.stderr_reader.join(2)
            try:
                self.assertTrue(stderr.closed, 'stderr reader must close its own captured stream before retry')
            finally:
                engine.close(timeout=.01)
        self.assertFalse(closer.is_alive())
        self.assertFalse(engine.stderr_reader.is_alive())
        self.assertTrue(stderr.closed)
        self.assertTrue(proc.stdin.closed and proc.stdout.closed)
        self.assertIsNone(engine.proc)
        self.assertFalse(engine.termination_uncertain)


    def test_confirmed_exit_closes_stdin_before_waiting_for_stderr_eof(self):
        read_fd, write_fd = os.pipe()
        entered = threading.Event()
        class Pipe(io.TextIOWrapper):
            def readline(self, *args):
                entered.set()
                return super().readline(*args)
        stderr = Pipe(os.fdopen(read_fd, 'rb'), encoding='utf-8')
        stdin = os.fdopen(write_fd, 'w')
        proc = mock.Mock(stdin=stdin, stdout=io.StringIO(), stderr=stderr)
        proc.poll.return_value = 0
        engine = StrataEngine('fixture', [], lazy=True)
        engine.proc = proc
        engine.stderr_reader = threading.Thread(target=engine._stderr_pump, args=(proc,), daemon=True)
        engine.stderr_reader.start()
        try:
            self.assertTrue(entered.wait(1))
            engine.close(timeout=.01)
            self.assertIsNone(engine.proc)
            self.assertTrue(stdin.closed and stderr.closed and proc.stdout.closed)
            self.assertFalse(engine.stderr_reader.is_alive())
        finally:
            stdin.close()
            engine.stderr_reader.join(2)
            engine.close(timeout=.01)


class ListenerLifetime(unittest.TestCase):
    def test_worker_start_failure_closes_listener_and_telemetry(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "unused"), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        http = mock.Mock()
        def close():
            svc.stopping.set()
            svc.telemetry.close()
        http.server_close.side_effect = close
        reader = mock.Mock()
        reader.start.side_effect = RuntimeError("fixture worker refused")
        svc.telemetry = mock.Mock()
        with mock.patch.object(server, "Server", return_value=http), \
                mock.patch.object(svc, "start_telemetry"), \
                mock.patch.object(server.threading, "Thread", return_value=reader):
            with self.assertRaisesRegex(RuntimeError, "fixture worker refused"):
                serve(svc)
        http.server_close.assert_called_once_with()
        svc.telemetry.close.assert_called_once_with()
        self.assertTrue(svc.stopping.is_set())

    def test_main_worker_failure_closes_owned_engine(self):
        tok = ByteTokenizer()
        engine = ResidentEngine(tok)
        engine.restart()
        with mock.patch.object(sys, "argv", ["server", "--engine", "mock", "--port", "0"]), \
                mock.patch.object(server, "MockEngine", return_value=engine), \
                mock.patch.object(server, "serve", side_effect=RuntimeError("fixture worker refused")):
            with self.assertRaisesRegex(RuntimeError, "fixture worker refused"):
                server.main()
        self.assertEqual(engine.closes, 1)
        self.assertFalse(engine.alive())

    def test_occupied_port_refuses_before_engine_and_never_falls_back(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with mock.patch.object(sys, "argv", ["server", "--engine", "mock", "--port", str(port)]), \
                    mock.patch.object(server, "MockEngine", side_effect=AssertionError("engine started before port check")), \
                    mock.patch.object(sys, "stderr", io.StringIO()) as err:
                with self.assertRaises(SystemExit) as caught:
                    server.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn(f"port {port} is already in use", err.getvalue())
            self.assertEqual(listener.getsockname()[1], port)

    def test_final_bind_race_closes_owned_engine_and_preserves_winning_listener(self):
        real_popen = server.popen
        code = "import sys; print('READY 4096 stop', flush=True); next(line for line in sys.stdin if line.strip() == 'QUIT')"
        with mock.patch.object(server, "popen", side_effect=lambda what, argv, **kw:
                               real_popen(what, [sys.executable, "-B", "-u", "-c", code], **kw)):
            engine = StrataEngine("fixture", [])
        child = engine.proc
        engine.close = mock.Mock(wraps=engine.close)
        self.addCleanup(engine.close)
        winner = socket.socket()
        self.addCleanup(winner.close)
        port = []
        def claim_port(svc, host, port):
            winner.bind((host, port))
            winner.listen()
            return serve(svc, host, port)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with mock.patch.object(sys, "argv", ["server", "--engine", "mock", "--port", str(port)]), \
                mock.patch.object(server, "MockEngine", return_value=engine), \
                mock.patch.object(server, "serve", side_effect=claim_port), \
                mock.patch.object(sys, "stderr", io.StringIO()) as err:
            with self.assertRaises(SystemExit) as caught:
                server.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(f"port {port} is already in use", err.getvalue())
        engine.close.assert_called_once_with()
        self.assertIsNotNone(child.poll())
        self.assertIsNone(engine.proc)
        self.assertTrue(child.stdin.closed)
        self.assertTrue(child.stdout.closed)
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass

    def test_main_shutdown_releases_listener_and_idle_worker(self):
        servers = []
        def ready(svc, host, port):
            http = serve(svc, host, port)
            servers.append(http)
            return http
        with mock.patch.object(sys, "argv", ["server", "--engine", "mock", "--port", "0"]), \
                mock.patch.object(server, "serve", side_effect=ready), \
                mock.patch.object(server.signal, "signal"), \
                mock.patch.object(server.time, "sleep", side_effect=KeyboardInterrupt):
            self.assertEqual(server.main(), 0)
        self.assertEqual(len(servers), 1)
        self.assertEqual(servers[0].socket.fileno(), -1)
        self.assertTrue(servers[0].svc.stopping.is_set())
        self.assertTrue(servers[0].svc.telemetry._stop.is_set())


if __name__ == "__main__":
    unittest.main()
