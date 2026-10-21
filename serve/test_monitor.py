"""Request inspection through HTTP; no native model or GPU is required."""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve import server
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve


class LaunchMemory(unittest.TestCase):
    def sample(self):
        return {"ram_gib": 32, "vram_gib": 16, "available_gib": 27, "commit_available_gib": None,
                "available_fallback_gib": None, "kv_ram_gib": 1.5, "cpu_server_gib": 2,
                "later_startup_gib": 1, "headroom_gib": 4, "vram_reserve_requested_mib": 1024,
                "vram_reserve_mib": 1024, "gpu_commit_gib": 0, "expert_pool_gib": 18.5,
                "pinned_gib": 12, "recent_mib": 6656, "windows_commit": False,
                "automatic_host_cache": True, "low_memory": False, "over_budget": False}

    def test_numeric_sample_allowlist_rejects_bad_types_nonfinite_and_missing_values(self):
        good = self.sample()
        raw = {**good, "private": "secret", "args": ["--pack", "private-path"], "env": {"API_KEY": "secret"}}
        evidence = server.launch_memory_evidence({"memory_policy": {"sample": raw}}, [], {}, automatic=True)
        self.assertEqual(evidence["sample"], {
            "ram_gib": 32, "vram_gib": 16, "available_gib": 27, "commit_available_gib": None,
            "available_fallback_estimate_gib": None, "kv_ram_allowance_gib": 1.5, "cpu_server_allowance_gib": 2,
            "later_startup_allowance_gib": 1, "headroom_requested_gib": 4, "vram_reserve_requested_mib": 1024,
            "vram_reserve_effective_request_mib": 1024, "gpu_commit_allowance_gib": 0, "expert_pool_budget_gib": 18.5,
            "pinned_budget_gib": 12, "recent_target_mib": 6656, "windows_commit": False,
            "automatic_host_cache": True, "low_memory": False, "over_budget": False})
        self.assertEqual(evidence["sample_status"], "available")
        for key, value in (("ram_gib", True), ("pinned_gib", "12"), ("vram_gib", float("nan")),
                           ("available_gib", float("inf")), ("recent_mib", -1), ("recent_mib", 10**500),
                           ("windows_commit", 1), ("automatic_host_cache", "true"),
                           ("headroom_gib", {"secret": "private"}), ("vram_reserve_mib", None)):
            with self.subTest(key=key, value=value):
                malformed = {**good, key: value}
                got = server.launch_memory_evidence({"memory_policy": {"sample": malformed}}, [], {}, automatic=True)
                self.assertEqual((got["sample_status"], got["sample"]), ("unavailable", None))
                json.dumps(got, allow_nan=False)
        for key in good:
            with self.subTest(missing=key):
                malformed = dict(good)
                del malformed[key]
                got = server.launch_memory_evidence({"memory_policy": {"sample": malformed}}, [], {}, automatic=True)
                self.assertIsNone(got["sample"])
        static = server.launch_memory_evidence({"memory_policy": {"sample": good}}, [], {}, automatic=False)
        self.assertEqual((static["cohort"], static["sample_status"], static["sample"]), ("static", "unavailable", None))

    def test_requested_controls_are_bounded_and_keep_unknown_values_unavailable(self):
        args = ["--max-context", "131072", "--resident-budget-gib", "private-value", "--kv", "private-path",
                "--expert-cache", "auto", "--prefill", "auto", "--pack", "private-path", "--no-prefill-borrow"]
        env = {"STRATA_RESIDENT_HEADROOM_GIB": "NaN", "STRATA_FILE_CACHE_MIB": "1" * 10000,
               "STRATA_FILE_CACHE_COMPACT": "private-value", "STRATA_OWNED_PRICE": "exact",
               "CUDA_MODULE_LOADING": "EAGER", "STRATA_API_KEY": "secret", "MODEL_PATH": "private-path"}
        got = server.launch_memory_evidence({}, args, env, automatic=False)
        self.assertEqual(got["requested_controls"], {
            "args": {"--max-context": 131072, "--resident-budget-gib": None, "--kv": None,
                     "--expert-cache": "auto", "--prefill": "auto", "--no-prefill-borrow": True},
            "env": {"STRATA_RESIDENT_HEADROOM_GIB": None, "STRATA_FILE_CACHE_MIB": None,
                    "STRATA_FILE_CACHE_COMPACT": None, "STRATA_OWNED_PRICE": "exact", "CUDA_MODULE_LOADING": "EAGER"}})
        for flags in (["--max-context"], ["--max-context", "1", "--max-context", "2"],
                      ["--max-context", "1e9999"], ["--max-context", True]):
            with self.subTest(flags=flags):
                got = server.launch_memory_evidence({}, flags, {}, automatic=False)
                self.assertIsNone(got["requested_controls"]["args"]["--max-context"])
        exact = server.launch_memory_evidence({}, ["--kv-resident", "9223372036854775807"], {}, automatic=False)
        self.assertEqual(exact["requested_controls"]["args"]["--kv-resident"], 9223372036854775807)
        self.assertLess(len(json.dumps(got, allow_nan=False)), 2048)

    def test_requested_environment_uses_windows_and_posix_name_semantics(self):
        env = {"strata_file_cache_mib": "2048", "STRATA_RESIDENT_HEADROOM_GIB": "4",
               "Strata_Resident_Headroom_Gib": "private-value", "Unknown_Case": "private-value"}
        for platform, expected in (("nt", {"STRATA_FILE_CACHE_MIB": 2048, "STRATA_RESIDENT_HEADROOM_GIB": None}),
                                   ("posix", {"STRATA_RESIDENT_HEADROOM_GIB": 4})):
            with self.subTest(platform=platform), mock.patch.object(server.os, "name", platform):
                got = server.launch_memory_evidence({}, [], env, automatic=False)
                self.assertEqual(got["requested_controls"]["env"], expected)

    def test_engine_argument_values_cannot_become_selected_options(self):
        import re
        source = (Path(__file__).resolve().parents[1] / "src/program/generate.cpp").read_text(encoding="utf-8")
        start = source.index("    Options o;")
        loop = source[start:source.index("#if defined(STRATA_USE_HIP)", start)]
        valued = set(re.findall(r'next\("([^"]+)"\)', loop)) | {
            "--expert-cache-device1", "--expert-cache-device2", "--expert-cache-device3",
            "--conversation-cache-mib", "--conversation-cache-min-free-mib",
            "--conversation-cache-slots", "--session-min-free-mib"}
        switches = set(re.findall(r'\ba\s*==\s*"([^"]+)"', loop)) - valued - {"--help", "-h"}
        for flag in valued:
            values = ["--no-prefill-borrow"] * (2 if flag == "--control-vector-layer-range" else 1)
            with self.subTest(value_of=flag):
                controls = server.launch_memory_evidence({}, [flag, *values], {}, automatic=False)["requested_controls"]["args"]
                self.assertNotIn("--no-prefill-borrow", controls)
        for flag in switches:
            with self.subTest(switch=flag):
                controls = server.launch_memory_evidence({}, [flag, "--max-context", "131072"], {}, automatic=False)["requested_controls"]["args"]
                self.assertEqual(controls["--max-context"], 131072)
        for flags in (["--unknown", "--no-prefill-borrow", "--max-context", "131072"],
                      ["--help", "--max-context", "131072"]):
            controls = server.launch_memory_evidence({}, flags, {}, automatic=False)["requested_controls"]["args"]
            self.assertIsNone(controls["--max-context"])
        for flags, selected in ((["--pack", "--no-prefill-borrow"], False),
                                (["--pack", "normal", "--no-prefill-borrow"], True),
                                (["--native", "--max-context", "--max-context", "131072"], False)):
            controls = server.launch_memory_evidence({}, flags, {}, automatic=False)["requested_controls"]["args"]
            self.assertEqual(controls.get("--no-prefill-borrow", False), selected)

    def test_integer_controls_never_use_floating_point_or_native_prefix_interpretation(self):
        cases = (("--max-context", None, 9223372036854775807),
                 ("--kv-resident", None, 9223372036854775807),
                 ("--prefill", None, 9223372036854775807),
                 ("--expert-cache", None, 2147483647),
                 ("--vram-reserve-mib", None, 2147483647),
                 (None, "STRATA_FILE_CACHE_MIB", 17592186044415))
        for flag, key, maximum in cases:
            for spelling, expected in (("0", 0), ("32768", 32768), ("00027", 27), (str(maximum), maximum),
                                       (str(maximum + 1), None),
                                       ("9007199254740993", 9007199254740993 if flag in
                                        ("--max-context", "--kv-resident", "--prefill") else None)):
                with self.subTest(flag=flag, key=key, spelling=spelling):
                    got = server.launch_memory_evidence({}, [flag, spelling] if flag else [],
                                                        {key: spelling} if key else {}, automatic=False)
                    result = got["requested_controls"]["args" if flag else "env"][flag or key]
                    self.assertEqual(result, expected)
                    if expected is not None:
                        self.assertIs(type(result), int)
            for spelling in ("32768e1", "32768.0", "32768.9", "9007199254740993.0", "9007199254740993e0",
                             "+32768", "-32768", " 32768", "32768 ", "\t32768", "32768\n", "0x8000",
                             "32768private", "32_768", "٣٢", "32\u00a0", "", "NaN", "inf", "1" * 65,
                             True, 32768, 32768.0, float("nan"), float("inf"), None):
                with self.subTest(flag=flag, key=key, spelling=spelling):
                    got = server.launch_memory_evidence({}, [flag, spelling] if flag else [],
                                                        {key: spelling} if key else {}, automatic=False)
                    self.assertIsNone(got["requested_controls"]["args" if flag else "env"][flag or key])
                    json.dumps(got, allow_nan=False)

    def test_real_controls_follow_atof_decimal_meaning_without_claiming_byte_capacity(self):
        for spelling, expected in (("3", 3.0), ("3.5", 3.5), ("35e-1", 3.5), ("3.5E+0", 3.5),
                                   ("+3.5", 3.5), (" \t3.5\r\n", 3.5), (".5", 0.5), ("3.", 3.0),
                                   ("9007199254740993", 9007199254740992.0),
                                   ("9007199254740993.0", 9007199254740992.0),
                                   ("9007199254740993e0", 9007199254740992.0)):
            with self.subTest(spelling=spelling):
                got = server.launch_memory_evidence({}, ["--resident-budget-gib", spelling],
                                                    {"STRATA_RESIDENT_HEADROOM_GIB": spelling}, automatic=False)
                for result in (got["requested_controls"]["args"]["--resident-budget-gib"],
                               got["requested_controls"]["env"]["STRATA_RESIDENT_HEADROOM_GIB"]):
                    self.assertEqual(result, expected)
                    self.assertIs(type(result), float)
        for spelling in ("-3.5", "3.5private", "0x1p2", "3_500", "\u00a03.5", "٣.٥", "NaN", "inf", "1e9999",
                         "1e20", "", True, 3, 3.5, None, float("nan"), float("inf")):
            with self.subTest(spelling=spelling):
                got = server.launch_memory_evidence({}, ["--resident-budget-gib", spelling],
                                                    {"STRATA_RESIDENT_HEADROOM_GIB": spelling}, automatic=False)
                self.assertIsNone(got["requested_controls"]["args"]["--resident-budget-gib"])
                self.assertIsNone(got["requested_controls"]["env"]["STRATA_RESIDENT_HEADROOM_GIB"])
                json.dumps(got, allow_nan=False)
        for spelling in ("0", "-0", "1e-9999"):
            with self.subTest(zero=spelling):
                got = server.launch_memory_evidence({}, ["--resident-budget-gib", spelling],
                                                    {"STRATA_RESIDENT_HEADROOM_GIB": spelling}, automatic=False)
                self.assertIsNone(got["requested_controls"]["args"]["--resident-budget-gib"])
                self.assertEqual(got["requested_controls"]["env"]["STRATA_RESIDENT_HEADROOM_GIB"], 0.0)

    def test_integer_auto_and_other_named_controls_keep_only_exact_safe_enums(self):
        got = server.launch_memory_evidence({}, ["--expert-cache", "auto", "--prefill", "auto", "--kv", "q4"],
                                            {"STRATA_FILE_CACHE_COMPACT": "1", "STRATA_OWNED_PRICE": "exact",
                                             "CUDA_MODULE_LOADING": "EAGER"}, automatic=False)
        self.assertEqual(got["requested_controls"], {
            "args": {"--expert-cache": "auto", "--prefill": "auto", "--kv": "q4"},
            "env": {"STRATA_FILE_CACHE_COMPACT": "1", "STRATA_OWNED_PRICE": "exact", "CUDA_MODULE_LOADING": "EAGER"}})
        for spelling in ("auto:16384", "AUTO", "1.0", "1e0", "+1", " 1", True):
            with self.subTest(spelling=spelling):
                got = server.launch_memory_evidence({}, ["--prefill", spelling],
                                                    {"STRATA_FILE_CACHE_COMPACT": spelling}, automatic=False)
                self.assertIsNone(got["requested_controls"]["args"]["--prefill"])
                self.assertIsNone(got["requested_controls"]["env"]["STRATA_FILE_CACHE_COMPACT"])

    def test_http_metrics_keep_snapshot_and_existing_key_and_host_checks(self):
        tok = ByteTokenizer()
        raw = self.sample()
        evidence = server.launch_memory_evidence({"memory_policy": {"sample": raw}}, [], {}, automatic=True)
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"),
                      launch_memory=evidence)
        svc.api_key = "test-key"
        with mock.patch.object(svc, "start_telemetry"):
            httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        raw["available_gib"] = 999
        evidence["sample"]["available_gib"] = 888
        try:
            for headers, status in (({}, 401), ({"Authorization": "Bearer wrong"}, 401),
                                    ({"Authorization": "Bearer test-key", "Host": "foreign.invalid"}, 200),
                                    ({"Authorization": "Bearer test-key"}, 200)):
                req = urllib.request.Request(base + "/metrics", headers=headers)
                try:
                    response = urllib.request.urlopen(req, timeout=10)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    body = json.loads(response.read())
                    self.assertEqual(response.status, status)
                    if status == 200:
                        self.assertEqual(body["launch_memory"]["sample"]["available_gib"], 27)
                    else:
                        self.assertNotIn("launch_memory", body)
            svc.api_key = ""
            with self.assertRaises(urllib.error.HTTPError) as denied:
                urllib.request.urlopen(urllib.request.Request(base + "/metrics", headers={"Host": "foreign.invalid"}),
                                       timeout=10)
            self.assertEqual(denied.exception.code, 403)
            denied.exception.close()
            reply = svc.metrics()
            reply["launch_memory"]["sample"]["available_gib"] = 777
            self.assertEqual(svc.metrics()["launch_memory"]["sample"]["available_gib"], 27)
        finally:
            httpd.shutdown()
            httpd.server_close()


class ReloadableEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded = True, False

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False


class MonitorOff(unittest.TestCase):
    """#332 is opt-in: by default no prompt or answer is kept, and the monitor's endpoints do not exist."""

    def test_off_by_default(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "secret"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
            self.assertEqual(len(svc.api_requests), 0)
            for path in ("/api/requests", "/api-monitor"):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base + path, timeout=10)
                self.assertEqual(e.exception.code, 404, path)
                e.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


class Monitor(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, "Thought.</think>\nHello <script>alert(1)</script>.", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True                       # opt-in ("api_monitor": true / --api-monitor)
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
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def chat(self, api="openai", **kwargs):
        # Anthropic thinks only when asked (#278): "thinking" there, reasoning_effort for OpenAI
        think = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if api == "anthropic" else             {"reasoning_effort": "low"}
        return self.request("/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Say hello."}],
                             "max_tokens": 256, **think, **kwargs})

    def latest(self):
        summary = self.request("/api/requests")[1]["requests"][0]
        return self.request("/api/requests?id=" + summary["id"])[1]

    def test_output_and_reasoning_for_both_dialects_and_stream_modes(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    code, reply = self.chat(api, stream=stream)
                    self.assertEqual(code, 200)
                    record = self.latest()
                    self.assertEqual(record["state"], "completed")
                    self.assertEqual(record["http_status"], 200)
                    self.assertIn("Say hello.", record["input"])
                    self.assertIn("<script>alert(1)</script>", record["output"])
                    self.assertIn("Thought.", record["reasoning"])
                    self.assertGreater(record["usage"].get("completion_tokens", record["usage"].get("output_tokens", 0)), 0)
                    self.assertGreaterEqual(record["wallclock_s"], record["first_token_s"])
                    self.assertNotIn("_clock", record)
                    if not stream:
                        self.assertEqual(json.loads(record["response"]), reply)
                    else:
                        self.assertNotIn("response", record)

    def test_error_and_engine_death_are_not_successes(self):
        self.assertEqual(self.chat(max_tokens=20000)[0], 400)
        self.assertEqual(self.latest()["state"], "error")
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream), mock.patch.object(self.engine, "generate", side_effect=EngineDied("stopped")):
                    self.chat(api, stream=stream)
                    record = self.latest()
                    self.assertEqual(record["state"], "error")
                    self.assertIn("stopped", record["error"]["message"])
                    self.assertEqual(record["output"], "")
                    self.assertNotIn("timings", record)

    def test_history_is_authorized_and_does_not_capture_headers(self):
        self.svc.api_key = "secret"
        auth = {"Authorization": "Bearer secret"}
        self.assertEqual(self.request("/api/requests")[0], 401)
        self.assertEqual(self.request("/api/requests?id=missing")[0], 401)
        self.assertEqual(len(self.svc.api_requests), 0)
        self.assertEqual(self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, auth)[0], 200)
        summary = self.request("/api/requests", headers=auth)[1]["requests"][0]
        self.assertNotIn("input", summary)
        self.assertNotIn("output", summary)
        self.assertEqual(self.request("/api/requests?id=" + summary["id"])[0], 401)
        detail = self.request("/api/requests?id=" + summary["id"], headers=auth)[1]
        self.assertNotIn("secret", json.dumps(detail))

    def test_capture_is_bounded_and_eviction_returns_404(self):
        huge = "x" * 262200
        first = self.svc.begin_request("/v1/chat/completions", {"messages": [{"content": huge}]})
        self.assertEqual(len(first["input"]), 262144)
        self.assertTrue(first["input_truncated"])
        for _ in range(100):
            self.svc.begin_request("/v1/chat/completions", {"messages": []})
        self.assertEqual(len(self.request("/api/requests")[1]["requests"]), 100)
        self.assertEqual(self.request("/api/requests?id=" + first["id"])[0], 404)
        # Exercise output capture without making a mock engine emit 262K tokens.
        handler = object.__new__(self.httpd.RequestHandlerClass)
        handler.record = self.svc.api_requests[-1]
        chunk = {"choices": [{"delta": {"content": huge, "reasoning_content": huge}}]}
        list(handler._capture((item for item in [chunk]), "openai"))
        for field in ("output", "reasoning"):
            self.assertEqual(len(handler.record[field]), 262144)
            self.assertTrue(handler.record[field + "_truncated"])

    def test_queue_wait_and_loading_are_attributed_separately(self):
        self.engine.unload()
        entered, release = threading.Event(), threading.Event()
        original = self.engine.restart
        def restart():
            entered.set()
            self.assertTrue(release.wait(5))
            original()
        with mock.patch.object(self.engine, "restart", side_effect=restart), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.chat)
            self.assertTrue(entered.wait(5))
            second = pool.submit(self.chat)
            deadline = time.monotonic() + 5
            while len(self.svc.api_requests) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(self.svc.api_requests), 2)
            history = self.request("/api/requests")[1]
            self.assertEqual({r["state"] for r in history["requests"]}, {"queued", "loading"})
            self.assertFalse(history["loaded"])
            self.assertTrue(history["auto_load"])
            time.sleep(0.03)
            release.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        records = list(self.svc.api_requests)
        self.assertGreater(records[0]["load_s"], 0.02)
        self.assertGreater(records[1]["queue_s"], 0.02)
        self.assertLess(records[1]["load_s"], records[0]["load_s"])
        self.assertTrue(all(r["state"] == "completed" for r in records))

    def test_page_and_relative_assets_are_served(self):
        code, html = self.request("/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn('src="/web/monitor.js"', html)
        code, script = self.request("/web/monitor.js")
        self.assertEqual(code, 200)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        self.assertEqual(self.request("/web/../server.py")[0], 404)


class ConversationCacheCard(unittest.TestCase):
    """#596: /metrics' "conversation_cache": the parked conversations from the engine's log lines, the budget and
    slots from its INFO line, and how much of the prompts the cache gave back."""

    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=2 bytes=209715200\n"

    def test_the_log(self):
        from serve.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("an earlier run\n" + self.PARK, encoding="utf-8")
            start = log.stat().st_size                 # this run starts here: the earlier run's park is not counted
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), start)["parked"], 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata serve: something else\n" + self.PARK + self.RESTORE + "strata serve: conversation ca")
            st = c.poll(str(log), start)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"]), (2, 209715200, 1, 1))
            self.assertEqual((st["last_event"], st["last_tokens"]), ("restored", 4000))
            with open(log, "a", encoding="utf-8") as f:   # the cut line is read once it is whole
                f.write("che: dropped 1 superseded copy of this conversation; parked=1\n")
            self.assertEqual(c.poll(str(log), start)["parked"], 1)
            self.assertEqual(c.poll(str(log), log.stat().st_size)["parked"], 0)   # the engine started again
            self.assertEqual(c.poll(None, None)["parked"], 0)

    def test_metrics(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4}
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual((c["enabled"], c["budget_mib"], c["slots"], c["parked"], c["requests"]),
                             (True, 8192, 4, 0, 0))
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual(c["requests"], 1)
            self.assertGreater(c["last_prompt"], 0)
            engine.info = {}
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                self.assertFalse(json.loads(r.read())["conversation_cache"]["enabled"])   # off: the reuse lines only
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cc-card", "cc-slots-text", "cc-mem-text", "cc-facts", "cc-note"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)


if __name__ == "__main__":
    unittest.main()
