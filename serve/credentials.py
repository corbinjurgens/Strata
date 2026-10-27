"""Shared API credential grammar and private, file-backed credentials."""
from __future__ import annotations

import ctypes
import hmac
import os
from pathlib import Path
import secrets
import stat

MAX_KEY_BYTES = 4096


class CredentialError(ValueError):
    """A deliberately secret-free credential diagnostic."""


def key_list(value) -> list[str]:
    """Legacy strings separate keys with commas; list elements are literal keys."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(k) for k in value if str(k)]
    return [k.strip() for k in str(value).split(",") if k.strip()]


def api_key_of(value):
    """Normalize surrounding header whitespace while retaining legacy source types."""
    if isinstance(value, (list, tuple)):
        keys = [str(k).strip() for k in value]
        if not keys or any(not k for k in keys):
            raise CredentialError("an API key was given but it is empty")
        return keys
    key = "" if value is None else str(value)
    if key and not key_list(key):
        raise CredentialError("an API key was given but it is empty")
    return key.strip()


def key_matches(given: str, key) -> bool:
    """Compare every key, accepting clients' UTF-8 or Latin-1 header encoding."""
    raw = given.encode("latin-1", "replace")
    ok = False
    for k in key_list(key):
        ok |= hmac.compare_digest(raw, k.encode())
        try:
            ok |= hmac.compare_digest(raw, k.encode("latin-1"))
        except UnicodeEncodeError:
            pass
    return ok


def _key_bytes(key: str) -> bytes:
    data = key.encode("utf-8")
    if (not data or len(data) > MAX_KEY_BYTES or key != key.strip() or len(key.splitlines()) != 1 or
            any(ord(c) < 32 or ord(c) == 127 for c in key) or key.startswith("\ufeff")):
        raise CredentialError("Invalid API credential file.")
    return data


def _absolute(path) -> Path:
    if not isinstance(path, (str, os.PathLike)) or not os.path.isabs(path):
        raise CredentialError("API credential file must use an absolute path.")
    return Path(path)


class _WindowsSecurity:
    """Use native handles so a protected DACL exists at CREATE_NEW publication."""

    def __init__(self):
        from ctypes import wintypes as w
        self.w = w
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.adv = ctypes.WinDLL("advapi32", use_last_error=True)
        self.kernel.GetCurrentProcess.restype = w.HANDLE
        self.kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p,
                                           w.DWORD, w.DWORD, w.HANDLE]
        self.kernel.CreateFileW.restype = w.HANDLE
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.LocalFree.argtypes = [ctypes.c_void_p]
        self.adv.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)]
        self.adv.GetTokenInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                               w.DWORD, ctypes.POINTER(w.DWORD)]
        self.adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(w.LPWSTR)]
        self.adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            w.LPCWSTR, w.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.adv.GetSecurityInfo.argtypes = [w.HANDLE, ctypes.c_int, w.DWORD,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.adv.GetSecurityDescriptorControl.argtypes = [ctypes.c_void_p,
            ctypes.POINTER(w.WORD), ctypes.POINTER(w.DWORD)]
        self.adv.GetAce.argtypes = [ctypes.c_void_p, w.DWORD, ctypes.POINTER(ctypes.c_void_p)]

    def sid_text(self, sid):
        result = self.w.LPWSTR()
        if not self.adv.ConvertSidToStringSidW(sid, ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return result.value
        finally:
            self.kernel.LocalFree(ctypes.cast(result, ctypes.c_void_p))

    def user(self):
        token = self.w.HANDLE()
        if not self.adv.OpenProcessToken(self.kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            size = self.w.DWORD()
            self.adv.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
            buf = ctypes.create_string_buffer(size.value)
            if not self.adv.GetTokenInformation(token, 1, buf, size, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            return self.sid_text(ctypes.c_void_p.from_buffer(buf).value)
        finally:
            self.kernel.CloseHandle(token)

    def open(self, path, create=False):
        import msvcrt
        descriptor = ctypes.c_void_p()
        class Attributes(ctypes.Structure):
            _fields_ = [("length", self.w.DWORD), ("descriptor", ctypes.c_void_p),
                        ("inherit", self.w.BOOL)]
        try:
            attrs = None
            if create:
                sid = self.user()
                if not self.adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                        f"O:{sid}D:P(A;;FA;;;{sid})", 1, ctypes.byref(descriptor), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                attrs = Attributes(ctypes.sizeof(Attributes), descriptor, False)
            handle = self.kernel.CreateFileW(str(path), 0xC0020000 if create else 0x80020000,
                0 if create else 1, ctypes.byref(attrs) if attrs else None,
                1 if create else 3, 0x00200000, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                return msvcrt.open_osfhandle(handle, os.O_BINARY | (os.O_RDWR if create else os.O_RDONLY))
            except BaseException:
                self.kernel.CloseHandle(handle)
                if create:
                    path.unlink()
                raise
        finally:
            if descriptor:
                self.kernel.LocalFree(descriptor)

    def validate(self, fd):
        import msvcrt
        owner, acl, descriptor = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        if self.adv.GetSecurityInfo(msvcrt.get_osfhandle(fd), 1, 5, ctypes.byref(owner), None,
                                   ctypes.byref(acl), None, ctypes.byref(descriptor)):
            raise CredentialError("Unsafe API credential file permissions.")
        try:
            control, revision = self.w.WORD(), self.w.DWORD()
            if not self.adv.GetSecurityDescriptorControl(descriptor, ctypes.byref(control), ctypes.byref(revision)):
                raise CredentialError("Unsafe API credential file permissions.")
            sid = self.user()
            if not acl or not control.value & 0x1000 or self.sid_text(owner) != sid:
                raise CredentialError("Unsafe API credential file permissions.")
            # ACL header: revision, reserved, size, ACE count, reserved.
            count = ctypes.c_ushort.from_address(acl.value + 4).value
            if not count:
                raise CredentialError("Unsafe API credential file permissions.")
            for index in range(count):
                ace = ctypes.c_void_p()
                if not self.adv.GetAce(acl, index, ctypes.byref(ace)):
                    raise CredentialError("Unsafe API credential file permissions.")
                if (ctypes.c_ubyte.from_address(ace.value).value != 0 or
                        self.sid_text(ace.value + 8) != sid):
                    raise CredentialError("Unsafe API credential file permissions.")
        finally:
            self.kernel.LocalFree(descriptor)


def _validate(fd):
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise CredentialError("Unsafe API credential file.")
    if os.name == "nt":
        _WindowsSecurity().validate(fd)
    elif info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise CredentialError("Unsafe API credential file permissions.")


def _open_file(path, *, create=False):
    if os.name == "nt":
        return _WindowsSecurity().open(path, create=create)
    flags = (os.O_CREAT | os.O_EXCL | os.O_WRONLY if create else os.O_RDONLY | os.O_NONBLOCK)
    return os.open(path, flags | os.O_NOFOLLOW, 0o600)


def read_key_file(path) -> list[str]:
    path = _absolute(path)
    try:
        fd = _open_file(path)
        with os.fdopen(fd, "rb") as handle:
            _validate(handle.fileno())
            data = handle.read(MAX_KEY_BYTES + 3)
        if data.endswith(b"\r\n"):
            data = data[:-2]
        elif data.endswith(b"\n"):
            data = data[:-1]
        key = data.decode("utf-8")
        _key_bytes(key)
        return [key]
    except (OSError, ValueError, UnicodeError):
        raise CredentialError("Cannot read a valid private API credential file.") from None


def remove_created_key(path: Path, identity) -> None:
    """Rollback only the file this operation created, preserving any replacement."""
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
        raise CredentialError("API credential file changed during publication; preserved it.")
    path.unlink()


def create_key_file(path, key=None) -> os.stat_result | None:
    """Return a new file's rollback identity, or None for validated reuse."""
    path = _absolute(path)
    try:
        data = _key_bytes(key if key is not None else secrets.token_urlsafe(32))
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = _open_file(path, create=True)
        except FileExistsError:
            existing = read_key_file(path)
            if key is not None and existing != [key]:
                raise CredentialError("Existing API credential file does not match migration.")
            return None
        identity = None
        handle = None
        try:
            identity = os.fstat(fd)
            handle = os.fdopen(fd, "wb")
            with handle:
                _validate(handle.fileno())
                handle.write(data + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                if handle is None:
                    os.close(fd)
            finally:
                if identity is None:
                    path.unlink(missing_ok=True)
                else:
                    remove_created_key(path, identity)
            raise
        return identity
    except (OSError, ValueError, UnicodeError):
        raise CredentialError("Cannot create a private API credential file.") from None


def resolve_credentials(cfg, cli_key=None, *, environ=None):
    """CLI > environment > file > inline; absence alone disables authentication."""
    env = os.environ if environ is None else environ
    if cli_key is not None:
        value = cli_key
    elif "STRATA_API_KEY" in env:
        value = env["STRATA_API_KEY"]
    elif "api_key_file" in cfg:
        return read_key_file(cfg["api_key_file"])
    elif "api_key" in cfg:
        value = cfg["api_key"]
    else:
        return ""
    result = api_key_of(value)
    if not key_list(result):
        raise CredentialError("an API key was given but it is empty")
    return result
