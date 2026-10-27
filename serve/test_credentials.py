import os
import contextlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serve import credentials as cred, server
from serve.server import child_env


class ChildCredentialTests(unittest.TestCase):
    def test_inherited_and_configured_credentials_do_not_reach_engine(self):
        with patch.dict(os.environ, {"STRATA_API_KEY": "inherited-secret"}):
            env = child_env({"env": {"strata_api_key": "configured-secret"}})
        self.assertFalse(any(name.upper() == "STRATA_API_KEY" for name in env))
        self.assertNotIn("inherited-secret", env.values())
        self.assertNotIn("configured-secret", env.values())
        for cfg in ({}, {"vision": {"cuda_device": 1}}):
            self.assertNotIn("STRATA_API_KEY", server.vision_env(cfg, env))


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-pair-u1-credentials-r2-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key = self.root / "api.key"
        patcher = patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def private(self, data=b"file-key\n"):
        cred.create_key_file(self.key, "initial-key")
        self.key.write_bytes(data)
        return {"api_key_file": str(self.key), "api_key": "inline-key"}

    def test_precedence_and_explicit_empty_sources(self):
        cfg = self.private()
        self.assertEqual(cred.resolve_credentials(cfg, "cli-key", environ={"STRATA_API_KEY": "env-key"}), "cli-key")
        self.assertEqual(cred.resolve_credentials(cfg, environ={"STRATA_API_KEY": "env-key"}), "env-key")
        self.assertEqual(cred.resolve_credentials(cfg, environ={}), ["file-key"])
        self.assertEqual(cred.resolve_credentials({"api_key": "inline-key"}, environ={}), "inline-key")
        self.assertEqual(cred.resolve_credentials({}, environ={}), "")
        for empty in ("", " ", ", ,", [], [""]):
            with self.subTest(source="cli", empty=empty), self.assertRaises(cred.CredentialError):
                cred.resolve_credentials(cfg, empty, environ={"STRATA_API_KEY": "env-key"})
            with self.subTest(source="environment", empty=empty), self.assertRaises(cred.CredentialError):
                cred.resolve_credentials(cfg, environ={"STRATA_API_KEY": empty})
            with self.subTest(source="inline", empty=empty), self.assertRaises(cred.CredentialError):
                cred.resolve_credentials({"api_key": empty}, environ={})
        self.assertEqual(cred.resolve_credentials(cfg, "cli-key", environ={"STRATA_API_KEY": ""}), "cli-key")

    def test_legacy_grammar_and_literal_file_comma(self):
        self.assertEqual(cred.key_list(cred.resolve_credentials({"api_key": " a, b "}, environ={})), ["a", "b"])
        self.assertEqual(cred.resolve_credentials({"api_key": [" a,b ", "c"]}, environ={}), ["a,b", "c"])
        cfg = self.private(b"a,b\n")
        resolved = cred.resolve_credentials(cfg, environ={})
        self.assertEqual(resolved, ["a,b"])
        self.assertTrue(cred.key_matches("a,b", resolved))
        self.assertFalse(cred.key_matches("a", resolved))
        self.assertFalse(cred.key_matches("b", resolved))
        self.assertIs(server.api_key_of, cred.api_key_of)
        self.assertIs(server.key_list, cred.key_list)
        self.assertIs(server.key_matches, cred.key_matches)

    def test_file_grammar_bounds_and_secret_free_errors(self):
        cfg = self.private()
        for suffix in (b"", b"\n", b"\r\n"):
            self.key.write_bytes("utf8-秘密,key".encode() + suffix)
            self.assertEqual(cred.resolve_credentials(cfg), ["utf8-秘密,key"])
        self.key.write_bytes(b"x" * cred.MAX_KEY_BYTES + b"\r\n")
        self.assertEqual(len(cred.read_key_file(self.key)[0]), cred.MAX_KEY_BYTES)
        for data in (b"", b"\n", b"one\ntwo", b"one\n\n", b"one\r", b" key", b"key ",
                     b"\xff", b"\xef\xbb\xbfkey", b"key\0value", b"x" * (cred.MAX_KEY_BYTES + 1),
                     "one\u2028two".encode()):
            self.key.write_bytes(data)
            with self.subTest(length=len(data)), self.assertRaises(cred.CredentialError) as error:
                cred.resolve_credentials(cfg)
            self.assertEqual(str(error.exception), "Cannot read a valid private API credential file.")
            self.assertNotIn(str(self.key), str(error.exception))

    def test_missing_relative_directory_unreadable_and_reparse_refuse(self):
        for path in (str(self.key), "relative.key", str(self.root), "", None, ["key"]):
            with self.subTest(path=path), self.assertRaises(cred.CredentialError):
                cred.resolve_credentials({"api_key_file": path})
        self.private()
        with patch.object(cred, "_validate", side_effect=OSError("private-path-and-key")):
            with self.assertRaises(cred.CredentialError) as error:
                cred.read_key_file(self.key)
            self.assertNotIn("private-path-and-key", str(error.exception))
        with patch.object(cred, "_open_file", side_effect=PermissionError("private-path-and-key")):
            with self.assertRaises(cred.CredentialError) as error:
                cred.read_key_file(self.key)
            self.assertNotIn("private-path-and-key", str(error.exception))
        info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_file_attributes=0x400)
        with patch.object(cred.os, "fstat", return_value=info), self.assertRaises(cred.CredentialError):
            cred._validate(7)

    def test_posix_owner_and_mode_security_seam(self):
        for uid, mode, accepted in ((42, 0o600, True), (42, 0o400, True),
                                    (42, 0o644, False), (42, 0o660, False), (43, 0o600, False)):
            info = SimpleNamespace(st_mode=stat.S_IFREG | mode, st_uid=uid)
            with self.subTest(uid=uid, mode=mode), patch.object(cred.os, "name", "posix"), \
                    patch.object(cred.os, "getuid", return_value=42, create=True), \
                    patch.object(cred.os, "fstat", return_value=info):
                if accepted:
                    cred._validate(7)
                else:
                    with self.assertRaises(cred.CredentialError):
                        cred._validate(7)

    def test_create_reuse_and_failure_cleanup(self):
        self.assertTrue(cred.create_key_file(self.key, "literal,key"))
        original = self.key.read_bytes()
        self.assertFalse(cred.create_key_file(self.key))
        self.assertFalse(cred.create_key_file(self.key, "literal,key"))
        with self.assertRaises(cred.CredentialError):
            cred.create_key_file(self.key, "different-key")
        self.assertEqual(self.key.read_bytes(), original)
        for seam in ("_validate", "fstat", "fsync"):
            self.key.unlink(missing_ok=True)
            owner = cred if seam == "_validate" else cred.os
            with patch.object(owner, seam, side_effect=OSError("secret-in-error")), \
                    self.assertRaises(cred.CredentialError) as error:
                cred.create_key_file(self.key, "never-print-this-key")
            self.assertFalse(self.key.exists())
            self.assertNotIn("secret-in-error", str(error.exception))
            self.assertNotIn("never-print-this-key", str(error.exception))

    def test_real_private_file_platform_permissions(self):
        if os.name == "nt":
            opened = []
            real_open = cred._WindowsSecurity.open
            def at_publication(security, path, create=False):
                fd = real_open(security, path, create=create)
                if create:
                    security.validate(fd)
                    self.assertEqual(os.fstat(fd).st_size, 0)
                    opened.append(True)
                return fd
            with patch.object(cred._WindowsSecurity, "open", at_publication):
                cred.create_key_file(self.key, "fixture-only-key")
            self.assertEqual(opened, [True])
        else:
            cred.create_key_file(self.key, "fixture-only-key")
        self.assertEqual(cred.read_key_file(self.key), ["fixture-only-key"])
        if os.name == "nt":
            import ctypes
            security = cred._WindowsSecurity()
            fd = security.open(self.key)
            try:
                security.validate(fd)
            finally:
                os.close(fd)
            # Replace only this task-owned fixture's DACL with a broad grant.
            descriptor = ctypes.c_void_p()
            security.adv.SetFileSecurityW.argtypes = [security.w.LPCWSTR, security.w.DWORD, ctypes.c_void_p]
            self.assertTrue(security.adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                "D:P(A;;FA;;;WD)", 1, ctypes.byref(descriptor), None))
            try:
                self.assertTrue(security.adv.SetFileSecurityW(str(self.key), 0x80000004, descriptor))
            finally:
                security.kernel.LocalFree(descriptor)
            with self.assertRaises(cred.CredentialError):
                cred.read_key_file(self.key)
            print("Windows private-file ACL: protected user-only owner/DACL accepted; broad Everyone DACL refused.")
        else:
            self.assertEqual(stat.S_IMODE(self.key.stat().st_mode), 0o600)
            self.key.chmod(0o644)
            with self.assertRaises(cred.CredentialError):
                cred.read_key_file(self.key)
        before = self.key.read_bytes()
        with self.assertRaises(cred.CredentialError):
            cred.create_key_file(self.key)
        self.assertEqual(self.key.read_bytes(), before)

    def test_posix_open_is_exclusive_private_and_no_follow(self):
        posix = SimpleNamespace(**{**vars(os), "name": "posix", "O_NOFOLLOW": 0x20000, "O_NONBLOCK": 0x800})
        with patch.object(cred, "os", posix), patch.object(posix, "open", return_value=7) as opened:
            self.assertEqual(cred._open_file(self.key, create=True), 7)
            opened.assert_called_once_with(self.key, os.O_CREAT | os.O_EXCL | os.O_WRONLY | 0x20000, 0o600)
            opened.reset_mock()
            cred._open_file(self.key)
            opened.assert_called_once_with(self.key, os.O_RDONLY | 0x800 | 0x20000, 0o600)

    def test_key_write_failure_removes_only_new_file(self):
        real_fdopen = os.fdopen
        class FailingWrite:
            def __init__(self, fd, mode):
                self.handle = real_fdopen(fd, mode)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.handle.__exit__(*args)
            def fileno(self):
                return self.handle.fileno()
            def write(self, data):
                raise OSError("do-not-echo-credential")
        with patch.object(cred.os, "fdopen", side_effect=FailingWrite), \
                self.assertRaises(cred.CredentialError) as error:
            cred.create_key_file(self.key, "do-not-echo-credential")
        self.assertFalse(self.key.exists())
        self.assertNotIn("do-not-echo-credential", str(error.exception))

    def test_windows_creation_security_and_handle_conversion_fail_without_publication(self):
        if os.name != "nt":
            return
        import msvcrt
        for owner, name in ((cred._WindowsSecurity, "user"), (msvcrt, "open_osfhandle")):
            with self.subTest(seam=name), patch.object(owner, name, side_effect=OSError("fixture-secret")), \
                    self.assertRaises(cred.CredentialError) as error:
                cred.create_key_file(self.key, "fixture-secret")
            self.assertFalse(self.key.exists())
            self.assertNotIn("fixture-secret", str(error.exception))

    def test_rollback_preserves_replaced_key_file(self):
        cred.create_key_file(self.key, "owned-key")
        identity = self.key.stat()
        retained = self.root / "retained.key"
        self.key.rename(retained)
        cred.create_key_file(self.key, "replacement-key")
        with self.assertRaisesRegex(cred.CredentialError, "changed during publication"):
            cred.remove_created_key(self.key, identity)
        self.assertEqual(cred.read_key_file(self.key), ["replacement-key"])
        self.assertEqual(cred.read_key_file(retained), ["owned-key"])

    def test_server_refuses_bad_credentials_before_bind_tokenizer_or_engine(self):
        for cfg, cli in (({"api_key_file": "relative.key"}, []),
                         ({"api_key_file": str(self.key)}, []),
                         ({"api_key": "inline"}, ["--api-key="])):
            config = self.root / "profile.json"
            config.write_text(json.dumps(cfg))
            output = io.StringIO()
            with patch.object(sys, "argv", ["server.py", "--engine", "strata", "--config", str(config), *cli]), \
                    patch.object(server, "Server") as bind, patch.object(server, "StrataEngine") as engine, \
                    patch.object(server, "ByteTokenizer") as tokenizer, contextlib.redirect_stderr(output):
                self.assertEqual(server.main(), 2)
            bind.assert_not_called()
            engine.assert_not_called()
            tokenizer.assert_not_called()
            self.assertNotIn("inline", output.getvalue())
