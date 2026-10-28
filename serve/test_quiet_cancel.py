"""A disconnect reaches the real engine pipe during a quiet prompt read, before its first PP or token.

The native stand-in deliberately says nothing until STOP. This covers the boundary that a MockEngine watching
the cancel Event itself cannot test: HTTP disconnect -> Event -> STOP -> DONE/BADM -> the next request.
No GPU or model is used.
"""
import json
import io
from pathlib import Path
import queue
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.request
import urllib.error

from serve import server


FAKE = r'''import json, queue, sys, threading, time
from pathlib import Path
log = Path(sys.argv[sys.argv.index('--events') + 1])
batch = int(sys.argv[sys.argv.index('--batch') + 1])
commands, stop = queue.Queue(), threading.Event()
def note(event):
    with log.open('a') as f:
        f.write(json.dumps({'event': event, 'at': time.monotonic()}) + '\n')
def reader():
    for line in sys.stdin:
        line = line.strip()
        if line == 'STOP':
            note('stop')
            stop.set()
        else:
            commands.put(line)
    commands.put('QUIT')
threading.Thread(target=reader, daemon=True).start()
if batch:
    print(f'INFO batch_slots={batch}', flush=True)
print('READY 4096 stop', flush=True)
count = 0
while True:
    line = commands.get()
    if line == 'QUIT':
        break
    if not line.startswith(('GEN ', 'BGEN ')):
        continue
    count += 1
    stop.clear()
    note('request_' + str(count))
    fields = line.split()
    slot = int(fields[1]) if fields[0] == 'BGEN' else None
    if count == 1:
        # The test's cleanup also sends STOP, so even a failed assertion leaves no long-lived child.
        stop.wait(15)
        time.sleep(.15)  # STOP was received, but the current chunk has not finished yet: it must drain.
        print('DONE 0 128 1 0 cancel 0 0 0', flush=True)
    else:
        print('T 90\nDONE 1 20 1 1 stop', flush=True)
    if slot is not None:
        print(f'BADM {slot} 0', flush=True)
'''


class QuietPromptCancel(unittest.TestCase):
    def check_hangup(self, api, stream, batch, max_new=1):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, events = root / 'engine.py', root / 'events.jsonl'
            script.write_text(FAKE, encoding='utf-8')
            real_popen = server.subprocess.Popen
            with mock.patch.object(server.subprocess, 'Popen', lambda cmd, **kw:
                                   real_popen([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = server.StrataEngine('strata', ['--batch', str(batch), '--events', str(events)])
            service = server.Service(engine, server.ByteTokenizer(),
                                     server.ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
            http = server.serve(service, port=0)
            port = http.server_address[1]
            sock = None
            reply, errors = [], []
            follower = None
            try:
                body = {'model': 'm', 'stream': stream, 'max_tokens': max_new,
                        'messages': [{'role': 'user', 'content': 'a long prompt'}],
                        'reasoning_effort': 'none'}
                if api == 'responses':
                    body = {'model': 'm', 'stream': stream, 'max_output_tokens': max_new,
                            'input': 'a long prompt', 'reasoning': {'effort': 'none'}}
                data = json.dumps(body).encode()
                path = {'openai': '/v1/chat/completions', 'anthropic': '/v1/messages',
                        'responses': '/v1/responses'}[api]
                sock = socket.create_connection(('127.0.0.1', port), timeout=5)
                sock.sendall(f'POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n'
                             f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)

                def has(event):
                    return events.exists() and any(json.loads(row)['event'] == event
                                                   for row in events.read_text().splitlines())

                deadline = time.monotonic() + 3
                while not has('request_1') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(has('request_1'), 'the first request did not reach the native pipe')
                closed = time.monotonic()
                sock.close()
                sock = None

                def follow():
                    data = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}],
                                       'max_tokens': 1, 'reasoning_effort': 'none'}).encode()
                    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=data,
                                                 headers={'Content-Type': 'application/json'})
                    try:
                        with urllib.request.urlopen(req, timeout=15) as response:
                            reply.append(json.load(response))
                    except Exception as exc:
                        errors.append(exc)

                follower = threading.Thread(target=follow)
                follower.start()
                deadline = closed + 2.5
                while not has('stop') and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(has('stop'), 'disconnect must send STOP before a PP line or 10 s heartbeat')
                follower.join(timeout=2.5)
                self.assertFalse(follower.is_alive(), 'the cancelled request did not release the next request')
                self.assertEqual(errors, [])
                self.assertEqual(reply[0]['choices'][0]['message']['content'], 'Z')
                self.assertTrue(engine.alive(), 'cancellation must preserve the resident engine')
                self.assertEqual(reply[0]['usage']['completion_tokens'], 1)
                self.assertEqual(service.history[-1]['engine_generated'], 1,
                                 'the next request must record its own DONE, not the cancelled request')
                if batch:
                    self.assertFalse(any(engine.slot_busy), 'the cancelled admission must release its slot')
                else:
                    self.assertEqual(engine.last['finish'], 'stop', 'the next request must read its own DONE')
            finally:
                if sock is not None:
                    sock.close()
                engine._send('STOP')
                if follower is not None:
                    follower.join(timeout=3)
                http.shutdown()
                http.server_close()
                engine.close()

    def test_anthropic_stream(self):
        self.check_hangup('anthropic', True, 0)

    def test_batch_solo_stream(self):
        # A batch-enabled engine runs an isolated request with more than one output token through GEN first.
        self.check_hangup('anthropic', True, 2, max_new=2)

    def test_other_http_modes(self):
        for api in ('openai', 'anthropic', 'responses'):
            for stream in (False, True):
                for batch in (0, 2):
                    if (api, stream, batch) == ('anthropic', True, 0):
                        continue
                    with self.subTest(api=api, stream=stream, batch=batch):
                        self.check_hangup(api, stream, batch)


class QuietWaitPolicy(unittest.TestCase):
    def test_expired_silence_is_not_a_heartbeat(self):
        engine = object.__new__(server.StrataEngine)
        engine.proc = mock.Mock()
        engine.proc.poll.return_value = None
        engine.proc.stdin = io.StringIO()
        engine.lines = mock.Mock()
        engine.can_stop, engine.silence_s, engine.log_path = True, 10, None
        now = [0.0]

        def clock():
            now[0] += .001                 # time also passes between calculating a wait and starting it
            return now[0]

        def get(timeout):
            now[0] += timeout
            raise queue.Empty

        engine.lines.get.side_effect = get
        with mock.patch.object(server.time, 'monotonic', clock):
            gen = engine.generate([], 10, {}, threading.Event())
            with self.assertRaises(server.EngineSilent):
                next(gen)                  # must fail before yielding an overdue heartbeat to a slow consumer
        engine.proc.kill.assert_called_once()
        self.assertNotIn('STOP', engine.proc.stdin.getvalue())

    def test_polling_keeps_ten_second_heartbeats(self):
        for control in (False, True):
            with self.subTest(control=control):
                engine = object.__new__(server.StrataEngine)
                engine.proc = mock.Mock()
                engine.proc.poll.return_value = None
                engine.proc.stdin = io.StringIO()
                engine.lines = mock.Mock()
                engine.can_stop, engine.silence_s = True, 0
                engine._ctl_mode = 'solo'
                engine._send = mock.Mock()
                now = [0.0]
                waits = []

                def get(timeout):
                    waits.append(timeout)
                    if now[0] >= 21.0:
                        return 'DONE 0 1 1 0 length'
                    now[0] += timeout
                    raise queue.Empty

                engine.lines.get.side_effect = get
                cancel = threading.Event()
                with mock.patch.object(server.time, 'monotonic', lambda: now[0]):
                    gen = engine._control(cancel, lambda t: None) if control else engine.generate([1], 1, {}, cancel)
                    heartbeats = list(gen)
                self.assertEqual(heartbeats, [None, None], 'a quiet 21 seconds emits two heartbeats, not 42')
                self.assertTrue(all(0 < wait <= .5 for wait in waits), 'cancellation uses bounded blocking waits')
                engine._send.assert_not_called()
                self.assertNotIn('STOP', engine.proc.stdin.getvalue())


CANCEL_ENGINE = r'''import json, os, queue, sys, threading, time
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--events') + 1])
mode = sys.argv[sys.argv.index('--mode') + 1]
batch = int(sys.argv[sys.argv.index('--batch') + 1])
commands, stop = queue.Queue(), threading.Event()
def note(event):
    with root.open('a') as f:
        f.write(json.dumps({'event': event, 'at': time.monotonic(), 'pid': os.getpid()}) + '\n')
def reader():
    for line in sys.stdin:
        line = line.strip()
        if line == 'STOP' or line.startswith('BSTOP '):
            note(line)
            stop.set()
        else:
            commands.put(line)
    commands.put('QUIT')
threading.Thread(target=reader, daemon=True).start()
print('INFO batch_slots=' + str(batch), flush=True)
print('READY 4096 stop', flush=True)
note('ready')
marker = root.with_suffix('.started')
while True:
    line = commands.get()
    if line == 'QUIT':
        note('quit')
        break
    if not line.startswith(('GEN ', 'BGEN ')):
        continue
    slot = int(line.split()[1]) if line.startswith('BGEN ') else None
    if marker.exists():
        print('T 90\nDONE 1 20 1 1 stop', flush=True)
        if slot is not None:
            print(f'BADM {slot} 0', flush=True)
        note('answered')
        continue
    marker.touch()
    stop.clear()
    if slot is not None:
        # Admission completed; decoding is quiet until the slot's BSTOP.
        print(f'DONE 0 20 1 0 length\nBADM {slot} 1', flush=True)
    note('active')
    stop.wait(15)
    if mode == 'stalled':
        while True:
            time.sleep(1)  # deliberately ignores STOP, QUIT and EOF; exact-owned termination is required
    time.sleep(.15)
    if slot is None:
        print('DONE 0 20 1 0 cancel', flush=True)
    else:
        print(f'BDONE {slot} 0 cancel 1', flush=True)
    note('acknowledged')
'''


class CancellationAdmission(unittest.TestCase):
    def test_cancellation_during_control_acquisition_refuses_only_cancelled_waiters(self):
        class Process:
            def __init__(self):
                self.stdin, self.stdout, self.stderr = io.StringIO(), io.StringIO('READY 4096 stop\n'), None
                self.rc = None
            def poll(self): return self.rc
            def wait(self, timeout): self.rc = 0; return 0
            def terminate(self): self.rc = -1
            def kill(self): self.rc = -9
        class GateLock:
            def __init__(self):
                self.lock, self.entered = threading.Lock(), threading.Event()
                self.lock.acquire()
            def acquire(self, timeout):
                self.entered.set()
                return self.lock.acquire(timeout=timeout)
            def release(self): self.lock.release()
            def locked(self): return self.lock.locked()
        from serve import launch_runtime
        for cancelled in (0, 1, 2):
            with self.subTest(cancelled=cancelled), mock.patch.object(server, 'popen', side_effect=lambda *a, **k: Process()), \
                    mock.patch.object(server, 'contain'), mock.patch.object(launch_runtime, 'build_record', return_value=None), \
                    mock.patch.object(server.StrataEngine, '_pump', lambda self: None):
                engine = server.StrataEngine('fixture', [])
                engine.ctl = GateLock()
                result = []
                def waiter():
                    try:
                        admission = engine._take_control(threading.Event(), 10)
                        while True: next(admission)
                    except StopIteration as stop:
                        result.append(stop.value)
                    except Exception as error:
                        result.append(type(error))
                thread = threading.Thread(target=waiter)
                thread.start()
                try:
                    self.assertTrue(engine.ctl.entered.wait(1))
                    for _ in range(cancelled):
                        with self.assertRaises(server.EngineSilent):
                            engine._cancel_timeout(engine.proc, engine.gen)
                        engine.restart(tries=1)
                    if not cancelled:
                        engine.restart(tries=1)
                    engine.ctl.release()
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(result, [server.EngineDied] if cancelled else [True])
                    self.assertEqual(engine.ctl.locked(), not cancelled)
                    self.assertEqual((engine.waiting, engine.wait_lens), (0, []))
                finally:
                    if engine.ctl.locked(): engine.ctl.release()
                    thread.join(2)
                    engine.close()


    def test_service_waiter_stays_failed_after_successive_cancelled_generations(self):
        tok = server.ByteTokenizer()
        engine = server.MockEngine(tok, 'Z')
        engine.gen = 1
        svc = server.Service(engine, tok, server.ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        entered, resume = threading.Event(), threading.Event()
        class Gate:
            def __enter__(self):
                entered.set()
                if not resume.wait(3): raise AssertionError('fixture gate not released')
            def __exit__(self, *args): pass
        svc.fifo = Gate()
        result = []
        def waiter():
            try:
                list(svc.run([11], False, [], 1, {}, threading.Event()))
                result.append('generated')
            except Exception as error:
                result.append(type(error))
        thread = threading.Thread(target=waiter)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            engine.gen, engine.cancelled_gen = 3, 2
            resume.set()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, [server.EngineDied])
        finally:
            resume.set()
            thread.join(2)


class CancellationDeadline(unittest.TestCase):
    def wait_for(self, predicate, seconds=3):
        end = time.monotonic() + seconds
        while not predicate() and time.monotonic() < end:
            time.sleep(.01)
        self.assertTrue(predicate(), 'fixture condition did not arrive within its deadline')

    def check_cancel(self, batch, mode, uncertain=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, events = root / 'cancel.py', root / 'events.jsonl'
            script.write_text(CANCEL_ENGINE, encoding='utf-8')
            real_popen = server.popen
            children = []
            def spawn(what, argv, **kwargs):
                child = real_popen(what, [sys.executable, '-B', str(script), *argv[1:]], **kwargs)
                children.append(child)
                return child
            with mock.patch.object(server, 'popen', side_effect=spawn):
                engine = server.StrataEngine('fixture', ['--events', str(events), '--mode', mode, '--batch', str(batch)])
                service = server.Service(engine, server.ByteTokenizer(),
                                         server.ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
                http = server.serve(service, port=0)
                port = http.server_address[1]
                first = engine.proc
                sock, pending = None, None
                denied = []
                replies = []
                def rows():
                    return [json.loads(row) for row in events.read_text().splitlines()] if events.exists() else []
                def seen(name):
                    return any(row['event'] == name for row in rows())
                def request(path, body=None):
                    req = urllib.request.Request(f'http://127.0.0.1:{port}' + path,
                                                 data=json.dumps(body).encode() if body is not None else None,
                                                 headers={'Content-Type': 'application/json'})
                    try:
                        response = urllib.request.urlopen(req, timeout=15)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        return response.status, json.load(response)
                try:
                    body = {'messages': [{'role': 'user', 'content': 'quiet'}], 'max_tokens': 1,
                            'stream': True, 'reasoning_effort': 'none'}
                    data = json.dumps(body).encode()
                    sock = socket.create_connection(('127.0.0.1', port), timeout=3)
                    sock.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n'
                                 b'Content-Type: application/json\r\n' +
                                 f'Content-Length: {len(data)}\r\n\r\n'.encode() + data)
                    self.wait_for(lambda: seen('active'))
                    if batch:
                        self.wait_for(lambda: engine.slot_live[0] is not None and engine.slot_live[0]['state'] == 'decoding')
                    if uncertain:
                        for method in ('terminate', 'kill', 'wait'):
                            patch = mock.patch.object(first, method, side_effect=OSError('fixture denial'))
                            patch.start()
                            denied.append(patch)
                    if not batch and mode == 'stalled':
                        pending = threading.Thread(target=lambda: replies.append(request('/v1/chat/completions',
                            {'messages': [{'role': 'user', 'content': 'pending'}], 'max_tokens': 1,
                             'reasoning_effort': 'none'})))
                        pending.start()
                        self.wait_for(lambda: service.status['queued'] == 1)
                    closed = time.monotonic()
                    sock.close()
                    sock = None
                    command = 'BSTOP 0' if batch else 'STOP'
                    self.wait_for(lambda: seen(command), 2)
                    stop_at = next(row['at'] for row in rows() if row['event'] == command)
                    self.assertLess(stop_at - closed, 1.5)
                    if batch:
                        self.assertTrue(engine.slot_busy[0], 'receipt of BSTOP alone must never release the slot')
                    if mode == 'cooperative':
                        self.wait_for(lambda: seen('acknowledged'))
                        self.wait_for(lambda: not any(engine.slot_busy) if batch else not service.status['busy'])
                        settled = time.monotonic() - closed
                        self.assertLess(settled, 2.5)
                        self.assertIs(engine.proc, first)
                        self.assertIsNone(first.poll())
                    elif uncertain:
                        self.wait_for(lambda: engine.termination_uncertain, 8)
                        settled = time.monotonic() - closed
                        self.assertIs(engine.proc, first)
                        self.assertIsNone(first.poll())
                        self.assertFalse(engine.alive())
                        if batch:
                            self.assertTrue(engine.slot_busy[0], 'unconfirmed termination must retain the occupied slot')
                        status = request('/v1/status')[1]
                        self.assertEqual((status['lifecycle'], status['loaded']), ('failed', False))
                        self.assertEqual(request('/v1/models')[1]['data'][0]['status']['value'], 'unloaded')
                        self.assertEqual(request('/v1/load', {})[1]['error']['code'], 'engine_died')
                        self.assertEqual(len(children), 1, 'uncertain exit must forbid replacement')
                    else:
                        self.wait_for(lambda: first.poll() is not None, 8)
                        self.wait_for(lambda: not any(engine.slot_busy) if batch else not service.status['busy'])
                        settled = time.monotonic() - closed
                        self.assertGreaterEqual(settled, 5.0, 'a stalled child must receive the full cooperative grace')
                        self.assertLess(settled, 8)
                        self.assertTrue(first.stdin.closed)
                        self.assertTrue(first.stdout.closed)
                    if pending is not None:
                        pending.join(3)
                        self.assertFalse(pending.is_alive())
                        self.assertEqual((replies[0][0], replies[0][1]['error']['code']), (503, 'engine_died'))
                    if not uncertain:
                        code, answer = request('/v1/chat/completions',
                            {'messages': [{'role': 'user', 'content': 'after'}], 'max_tokens': 1,
                             'reasoning_effort': 'none'})
                        self.assertEqual((code, answer['choices'][0]['message']['content']), (200, 'Z'))
                        self.assertEqual(len(children), 1 if mode == 'cooperative' else 2)
                        if mode == 'stalled':
                            self.assertIsNot(engine.proc, first)
                    print(f'cancel-fixture mode={mode} batch={batch} uncertain={uncertain} '
                          f'STOP={stop_at - closed:.3f}s settled={settled:.3f}s')
                finally:
                    for patch in reversed(denied):
                        patch.stop()
                    if sock is not None:
                        sock.close()
                    for child in children:
                        if child.poll() is None and mode == 'stalled':
                            child.kill()
                            child.wait(timeout=3)
                    http.shutdown()
                    http.server_close()
                    engine.close()
                    if pending is not None:
                        pending.join(3)
                self.assertTrue(all(child.poll() is not None for child in children), 'fixture child leaked')

    def test_cooperative_solo_and_decoding_slot_acknowledge_before_reuse(self):
        for batch in (0, 2):
            with self.subTest(batch=batch):
                self.check_cancel(batch, 'cooperative')

    def test_nonresponsive_solo_and_decoding_slot_exit_before_replacement(self):
        for batch in (0, 2):
            with self.subTest(batch=batch):
                self.check_cancel(batch, 'stalled')

    def test_denied_termination_retains_owned_child_and_slot_and_fails_waiters(self):
        for batch in (0, 2):
            with self.subTest(batch=batch):
                self.check_cancel(batch, 'stalled', uncertain=True)


if __name__ == '__main__':
    unittest.main()
