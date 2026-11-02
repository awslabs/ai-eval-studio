# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``run_bounded_subprocess`` — the timeout, the group kill, the reap.

Every helper subprocess in the harness path goes through this one function, and
each of its three jobs was missing somewhere before it existed: ``uv sync`` and
the validation probe awaited ``communicate()`` with no timeout at all, and the two
call sites that did have one killed the immediate child without awaiting it.

The grandchild tests are the ones that matter. ``uv run python run_eval.py`` makes
the interpreter a grandchild of ``uv``, and the interpreter is the process calling
Bedrock — so killing only the child leaves the spend running. A test that asserted
"the helper raised" would have passed against that bug. They drive the helper
itself on both of its exits, the timeout and a cancelled caller, so they also pin
the ``start_new_session`` the group kill depends on.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

from app.services import harness_runner
from app.services.harness_runner import SubprocessTimeout, run_bounded_subprocess


def _pid_alive(pid: int) -> bool:
    """True when a process with this pid still exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but is not ours to signal; for this test that counts as alive.
        return True
    return True


def _spawner_argv(pid_file: Path) -> list[str]:
    """argv for a parent that starts a long-lived child and waits on it.

    The parent writes the child's pid to ``pid_file`` first, because its stdout
    is only returned to the caller on a clean exit — and these tests are about
    the exits that are not clean.
    """
    code = (
        "import pathlib, subprocess, sys;"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid));"
        "child.wait()"
    )
    return [sys.executable, "-c", code]


async def _read_pid_file(pid_file: Path, timeout_s: float = 10.0) -> int:
    """Wait for the spawner to record its child, and return that pid."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        text = pid_file.read_text().strip() if pid_file.exists() else ""
        if text:
            return int(text)
        if loop.time() > deadline:
            raise AssertionError("the spawner never recorded its child's pid")
        await asyncio.sleep(0.05)


async def _wait_until_gone(pid: int, timeout_s: float = 5.0) -> None:
    """Give a delivered signal time to land. Delivery to a group is not instant."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while _pid_alive(pid) and loop.time() < deadline:
        await asyncio.sleep(0.1)


class TestTheTimeout:
    @pytest.mark.asyncio
    async def test_a_wedged_process_raises_instead_of_hanging(self, tmp_path):
        with pytest.raises(SubprocessTimeout) as raised:
            await run_bounded_subprocess(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout_s=0.5,
                cwd=str(tmp_path),
                env=dict(os.environ),
                label="sleeper",
            )
        assert raised.value.label == "sleeper"
        assert raised.value.timeout_s == 0.5

    @pytest.mark.asyncio
    async def test_output_is_returned_when_the_process_finishes_in_time(self, tmp_path):
        returncode, stdout, stderr = await run_bounded_subprocess(
            [sys.executable, "-c", "print('done')"],
            timeout_s=30,
            cwd=str(tmp_path),
            env=dict(os.environ),
            label="quick",
        )
        assert returncode == 0
        assert stdout.strip() == b"done"
        assert stderr == b""

    @pytest.mark.asyncio
    async def test_a_nonzero_exit_is_returned_not_raised(self, tmp_path):
        """A failing command is the caller's business — each call site phrases its
        own error. Only a timeout raises."""
        returncode, _stdout, stderr = await run_bounded_subprocess(
            [sys.executable, "-c", "import sys; sys.stderr.write('nope'); sys.exit(3)"],
            timeout_s=30,
            cwd=str(tmp_path),
            env=dict(os.environ),
            label="failing",
        )
        assert returncode == 3
        assert stderr.strip() == b"nope"


class TestTheGroupKill:
    @pytest.mark.asyncio
    async def test_a_grandchild_does_not_survive_the_timeout(self, tmp_path, monkeypatch):
        """The parent spawns a long-lived child and waits on it. Killing only the
        parent would leave that child running — which in production is the
        interpreter making Bedrock calls, still billing after the turn gave up.

        Driven through ``run_bounded_subprocess`` rather than by calling the kill
        directly, so the spawn flags are part of what is tested: without
        ``start_new_session`` the pid leads no group of its own and only the
        parent dies. The grace period is shortened because what is under test is
        WHO gets killed, not how long the kill waits first.
        """
        real_kill = harness_runner.kill_harness_subprocess

        async def quick_kill(pid: int) -> None:
            await real_kill(pid, grace_seconds=0.2)

        monkeypatch.setattr(harness_runner, "kill_harness_subprocess", quick_kill)

        pid_file = tmp_path / "grandchild.pid"
        run = asyncio.ensure_future(
            run_bounded_subprocess(
                _spawner_argv(pid_file),
                timeout_s=3.0,
                cwd=str(tmp_path),
                env=dict(os.environ),
                label="spawner",
            )
        )
        grandchild_pid = await _read_pid_file(pid_file)
        assert _pid_alive(grandchild_pid), "the fixture's grandchild never started"

        with pytest.raises(SubprocessTimeout):
            await run

        await _wait_until_gone(grandchild_pid)
        assert not _pid_alive(grandchild_pid), (
            f"grandchild {grandchild_pid} outlived the timeout — in production "
            "this is a harness interpreter still calling Bedrock after the turn "
            "stopped waiting for it"
        )

    @pytest.mark.asyncio
    async def test_a_grandchild_does_not_survive_cancellation(self, tmp_path):
        """A cancelled turn or job stops awaiting the helper mid-run. The group has
        to die then too: the timeout is not the only way a caller stops waiting,
        and a group left running after a cancel spends exactly what the timeout
        path exists to stop."""
        pid_file = tmp_path / "grandchild.pid"
        run = asyncio.ensure_future(
            run_bounded_subprocess(
                _spawner_argv(pid_file),
                timeout_s=60,
                cwd=str(tmp_path),
                env=dict(os.environ),
                label="spawner",
            )
        )
        grandchild_pid = await _read_pid_file(pid_file)
        assert _pid_alive(grandchild_pid), "the fixture's grandchild never started"

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        await _wait_until_gone(grandchild_pid)
        assert not _pid_alive(grandchild_pid), (
            f"grandchild {grandchild_pid} outlived the cancellation — the helper "
            "stopped waiting but left the group running"
        )

    @pytest.mark.asyncio
    async def test_a_process_sharing_our_group_is_signalled_alone(self, tmp_path, monkeypatch):
        """A child spawned WITHOUT ``start_new_session`` shares this process's
        group, so ``killpg`` on its pgid would signal the test runner — in
        production, the coordinator. The kill has to fall back to the one process.

        ``killpg`` is replaced with a recorder so that a regression is reported
        rather than acted on.
        """
        group_signals: list[tuple[int, int]] = []
        monkeypatch.setattr(
            harness_runner.os, "killpg",
            lambda pgid, sig: group_signals.append((pgid, sig)),
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            cwd=str(tmp_path),
        )
        try:
            assert os.getpgid(proc.pid) != proc.pid, (
                "precondition: the child must share this process's group"
            )
            await harness_runner.kill_harness_subprocess(proc.pid, grace_seconds=0.2)

            assert group_signals == [], (
                f"signalled process group(s) {group_signals} for a pid that leads "
                "none — in production that group is the coordinator's own"
            )
            await asyncio.wait_for(proc.wait(), timeout=10)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    @pytest.mark.asyncio
    async def test_the_process_is_reaped_so_no_zombie_is_left(self, tmp_path):
        """``kill`` only asks; something has to collect the exit status. Two call
        sites killed and returned, leaving the event loop holding a transport for
        a process nobody would wait on."""
        with pytest.raises(SubprocessTimeout):
            await run_bounded_subprocess(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout_s=0.5,
                cwd=str(tmp_path),
                env=dict(os.environ),
                label="zombie-check",
            )

        # A surviving zombie would still answer signal 0 as its parent has not
        # collected it. Nothing here is left to collect, so the pid is gone.
        # Sampling every child of this process is the only portable check.
        try:
            reaped_pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            reaped_pid = 0  # no children at all, which is the expected state
        assert reaped_pid == 0, (
            f"pid {reaped_pid} was still waiting to be collected; the helper "
            "returned before reaping the process it killed"
        )
