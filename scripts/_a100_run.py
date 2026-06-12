"""Internal: drive the A100 SSH chain non-interactively via asyncssh.

This script is local-only tooling for the v1.2s3 run; it is not part of the
shipping library and stays under scripts/ rather than src/.

Usage:
  python scripts/_a100_run.py probe
  python scripts/_a100_run.py run "<remote shell command>"

It expects A100_PASSWORD in the environment so the password is never
committed. The SSH chain is jump host 10.115.7.6:25711 → inner 10.106.200.205:2222.
"""
from __future__ import annotations

import asyncio
import os
import sys

import asyncssh

JUMP_HOST = ("10.115.7.6", 25711, "root")
INNER_HOST = ("10.106.200.205", 2222, "root")


async def _open_inner(password: str) -> tuple[asyncssh.SSHClientConnection, asyncssh.SSHClientConnection]:
    jump = await asyncssh.connect(
        JUMP_HOST[0],
        port=JUMP_HOST[1],
        username=JUMP_HOST[2],
        password=password,
        known_hosts=None,
    )
    inner = await asyncssh.connect(
        INNER_HOST[0],
        port=INNER_HOST[1],
        username=INNER_HOST[2],
        password=password,
        tunnel=jump,
        known_hosts=None,
    )
    return jump, inner


async def probe(password: str) -> int:
    jump, inner = await _open_inner(password)
    try:
        r1 = await jump.run("hostname && nvidia-smi -L 2>/dev/null | head -3 || echo no-smi-jump")
        r2 = await inner.run("hostname && nvidia-smi -L 2>/dev/null | head -3 || echo no-smi-inner")
        print("== jump ==")
        print(r1.stdout)
        print("== inner ==")
        print(r2.stdout)
    finally:
        inner.close()
        jump.close()
    return 0


async def run(password: str, cmd: str) -> int:
    jump, inner = await _open_inner(password)
    try:
        result = await inner.run(cmd, check=False)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.exit_status if result.exit_status is not None else 1
    finally:
        inner.close()
        jump.close()


async def put(password: str, local: str, remote: str) -> int:
    """Stream a local file to a remote path in base64-encoded chunks via shell appends.

    asyncssh SFTP raises SFTPNoSuchFile against some upstream openssh builds, and
    a single stdin-piped command throws BrokenPipeError on larger payloads (the
    remote heredoc consumer closes before all bytes arrive), so we chunk through
    `printf %s ... | base64 -d >>`.
    """
    import base64

    with open(local, "rb") as fh:
        payload = fh.read()
    encoded = base64.b64encode(payload).decode("ascii")
    jump, inner = await _open_inner(password)
    try:
        await inner.run(
            f"REMOTE_TARGET={remote!s}; mkdir -p $(dirname \"$REMOTE_TARGET\") && : > \"$REMOTE_TARGET\""
        )
        chunk_size = 8 * 1024
        for offset in range(0, len(encoded), chunk_size):
            chunk = encoded[offset : offset + chunk_size]
            cmd = (
                f"REMOTE_TARGET={remote!s}; "
                f"printf '%s' {chunk!r} | base64 -d >> \"$REMOTE_TARGET\""
            )
            result = await inner.run(cmd, check=False)
            if result.exit_status:
                sys.stderr.write(result.stderr)
                return result.exit_status or 1
        check = await inner.run(
            f"REMOTE_TARGET={remote!s}; wc -c < \"$REMOTE_TARGET\"", check=False
        )
        print(f"uploaded {local} -> {remote} (remote bytes: {check.stdout.strip()})")
    finally:
        inner.close()
        jump.close()
    return 0


async def get(password: str, remote: str, local: str) -> int:
    """Stream a remote file to a local path via base64-over-ssh (mirrors put())."""
    import base64

    jump, inner = await _open_inner(password)
    try:
        cmd = f"REMOTE_TARGET={remote!s}; base64 -w 0 \"$REMOTE_TARGET\""
        result = await inner.run(cmd, check=False)
        if result.exit_status:
            sys.stderr.write(result.stderr)
            return result.exit_status or 1
        payload = base64.b64decode(result.stdout.strip())
        with open(local, "wb") as fh:
            fh.write(payload)
        print(f"downloaded {remote} -> {local} ({len(payload)} bytes)")
    finally:
        inner.close()
        jump.close()
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    password = os.environ.get("A100_PASSWORD")
    if not password:
        print("error: A100_PASSWORD env var must be set", file=sys.stderr)
        return 2
    mode = sys.argv[1]
    if mode == "probe":
        return asyncio.run(probe(password))
    if mode == "run":
        if len(sys.argv) < 3:
            print("usage: scripts/_a100_run.py run '<cmd>'", file=sys.stderr)
            return 2
        return asyncio.run(run(password, sys.argv[2]))
    if mode == "put":
        local = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("LOCAL_PATH", "")
        remote = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("REMOTE_PATH", "")
        if not local or not remote:
            print("usage: scripts/_a100_run.py put <local> <remote>", file=sys.stderr)
            return 2
        return asyncio.run(put(password, local, remote))
    if mode == "get":
        remote = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("REMOTE_PATH", "")
        local = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("LOCAL_PATH", "")
        if not remote or not local:
            print("usage: scripts/_a100_run.py get <remote> <local>", file=sys.stderr)
            return 2
        return asyncio.run(get(password, remote, local))
    print(f"unknown mode: {mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
