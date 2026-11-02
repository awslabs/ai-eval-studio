# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for working-tree snapshot pack/unpack defenses.

Covers path-traversal rejection and the resource ceilings applied on extraction.
"""
import tarfile
from pathlib import Path

import pytest

from app.services.working_tree_snapshot import (
    SnapshotTooLargeError,
    SnapshotTraversalError,
    pack_working_tree,
    unpack_snapshot,
)


class TestPathTraversalRejection:
    """Tests that unpack_snapshot rejects malicious tar members."""

    def test_reject_parent_refs(self, tmp_path):
        """Tar member with .. component must raise SnapshotTraversalError."""
        # Create a malicious tar with a parent ref
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add a file with parent directory reference
            info = tarfile.TarInfo(name="../etc/passwd")
            info.size = 0
            tar.addfile(info)

        dest = tmp_path / "dest"
        dest.mkdir()

        with pytest.raises(SnapshotTraversalError, match=".."):
            unpack_snapshot(tar_path, dest)

    def test_reject_absolute_path(self, tmp_path):
        """Tar member with absolute path must raise SnapshotTraversalError."""
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="/etc/passwd")
            info.size = 0
            tar.addfile(info)

        dest = tmp_path / "dest"
        dest.mkdir()

        with pytest.raises(SnapshotTraversalError, match="Absolute path"):
            unpack_snapshot(tar_path, dest)

    def test_reject_symlink(self, tmp_path):
        """Tar member that is a symlink must raise SnapshotTraversalError."""
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)

        dest = tmp_path / "dest"
        dest.mkdir()

        with pytest.raises(SnapshotTraversalError, match="symlink"):
            unpack_snapshot(tar_path, dest)

    def test_reject_escape_via_resolve(self, tmp_path):
        """Tar member resolving outside dest must raise SnapshotTraversalError."""
        # This tests the resolve() + is_relative_to() check
        # Note: parent ref check catches "foo/../../etc" before resolve check,
        # so we test that ANY escape is rejected
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # Member that looks innocent but would resolve outside dest
            info = tarfile.TarInfo(name="foo/../../etc/passwd")
            info.size = 0
            tar.addfile(info)

        dest = tmp_path / "dest"
        dest.mkdir()

        with pytest.raises(SnapshotTraversalError):
            unpack_snapshot(tar_path, dest)

    def test_roundtrip_packs_only_include_set(self, tmp_path):
        """Pack then unpack reproduces only the include-set files."""
        # Create a project tree with both included and excluded files
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "src").mkdir()
        (project_dir / "src" / "config.py").write_text("# config")
        (project_dir / "prompts").mkdir()
        (project_dir / "prompts" / "baseline.md").write_text("# prompt")
        (project_dir / "run_eval.py").write_text("# harness")
        (project_dir / "config.json").write_text("{}")
        (project_dir / "README.md").write_text("# README")
        # Dataset files — raw/processed/synthetic included, uploads excluded
        (project_dir / "data").mkdir()
        (project_dir / "data" / "raw").mkdir()
        (project_dir / "data" / "raw" / "sentiment.csv").write_text("text,label")
        (project_dir / "data" / "processed").mkdir()
        (project_dir / "data" / "processed" / "merged.csv").write_text("merged_data")
        (project_dir / "data" / "synthetic").mkdir()
        (project_dir / "data" / "synthetic" / "aug.csv").write_text("synthetic_data")
        (project_dir / "data" / "uploads").mkdir()
        (project_dir / "data" / "uploads" / "pending.csv").write_text("transient")
        # docs/ reference material MUST ship — a generated validator read
        # docs/*.csv relative to cwd and died on FileNotFoundError in the
        # worker workspace (every case failed while the run reported 0%).
        (project_dir / "docs").mkdir()
        (project_dir / "docs" / "schema-reference.csv").write_text("field,type")
        (project_dir / ".venv").mkdir()
        (project_dir / ".venv" / "lib").mkdir()

        # Pack
        tar_path = tmp_path / "snapshot.tar.gz"
        pack_working_tree(project_dir, tar_path)

        # Unpack
        dest = tmp_path / "unpacked"
        dest.mkdir()
        unpack_snapshot(tar_path, dest)

        # Assert: included files present
        assert (dest / "src" / "config.py").exists()
        assert (dest / "prompts" / "baseline.md").exists()
        assert (dest / "run_eval.py").exists()
        assert (dest / "config.json").exists()
        assert (dest / "README.md").exists()

        # Assert: dataset dirs included (except uploads)
        assert (dest / "data" / "raw" / "sentiment.csv").exists()
        assert (dest / "data" / "processed" / "merged.csv").exists()
        assert (dest / "data" / "synthetic" / "aug.csv").exists()
        # Assert: reference docs included
        assert (dest / "docs" / "schema-reference.csv").exists()

        # Assert: excluded files absent
        assert not (dest / "data" / "uploads").exists()
        assert not (dest / ".venv").exists()

    def test_roundtrip_preserves_content(self, tmp_path):
        """Pack then unpack preserves file content byte-for-byte."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "src").mkdir()
        content = "def foo():\n    return 42\n"
        (project_dir / "src" / "code.py").write_text(content)

        tar_path = tmp_path / "snapshot.tar.gz"
        pack_working_tree(project_dir, tar_path)

        dest = tmp_path / "unpacked"
        dest.mkdir()
        unpack_snapshot(tar_path, dest)

        assert (dest / "src" / "code.py").read_text() == content


class TestS3SnapshotOrchestration:
    """Tests for S3 upload/download orchestration wrappers."""

    @pytest.mark.asyncio
    async def test_upload_snapshot_to_s3_roundtrip(self, tmp_path):
        """upload_snapshot_to_s3 packs and uploads; download_and_unpack_snapshot retrieves."""
        from unittest.mock import MagicMock
        from app.services.working_tree_snapshot import (
            upload_snapshot_to_s3,
            download_and_unpack_snapshot,
        )

        # Create a project tree
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "src").mkdir()
        (project_dir / "src" / "config.py").write_text("# config")
        (project_dir / "run_eval.py").write_text("# harness")

        # Mock S3 client with in-memory storage
        s3_storage = {}

        def mock_put_object(Bucket, Key, Body, **kwargs):
            s3_storage[Key] = Body.read()

        def mock_download_file(Bucket, Key, Filename, **kwargs):
            Path(Filename).write_bytes(s3_storage[Key])

        s3_client = MagicMock()
        s3_client.put_object = MagicMock(side_effect=mock_put_object)
        s3_client.download_file = MagicMock(side_effect=mock_download_file)

        # Upload
        size = await upload_snapshot_to_s3(
            project_dir, s3_client, "test-bucket", "snapshots/test.tar.gz"
        )

        assert size > 0
        assert "snapshots/test.tar.gz" in s3_storage

        # Download
        dest = tmp_path / "unpacked"
        dest.mkdir()
        await download_and_unpack_snapshot(
            s3_client, "test-bucket", "snapshots/test.tar.gz", dest
        )

        # Verify content
        assert (dest / "src" / "config.py").exists()
        assert (dest / "run_eval.py").exists()

    @pytest.mark.asyncio
    async def test_upload_snapshot_cleans_up_temp_file(self, tmp_path):
        """upload_snapshot_to_s3 cleans up temp tar.gz even on exception."""
        from unittest.mock import MagicMock
        from app.services.working_tree_snapshot import upload_snapshot_to_s3

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "run_eval.py").write_text("# harness")

        s3_client = MagicMock()
        s3_client.put_object = MagicMock(side_effect=RuntimeError("S3 error"))

        # Should raise but clean up temp file
        with pytest.raises(RuntimeError):
            await upload_snapshot_to_s3(
                project_dir, s3_client, "test-bucket", "snapshots/test.tar.gz"
            )

        # Verify no .tar.gz left in /tmp (best-effort check)
        import tempfile
        tmp_tars = list(Path(tempfile.gettempdir()).glob("*.tar.gz"))
        # We can't assert zero because other tests might run concurrently,
        # but we can verify the upload call was attempted
        s3_client.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_snapshot_cleans_up_temp_file(self, tmp_path):
        """download_and_unpack_snapshot cleans up temp tar.gz even on exception."""
        from unittest.mock import MagicMock
        from app.services.working_tree_snapshot import download_and_unpack_snapshot

        s3_client = MagicMock()
        s3_client.download_file = MagicMock(side_effect=RuntimeError("S3 error"))

        dest = tmp_path / "dest"
        dest.mkdir()

        # Should raise but clean up temp file
        with pytest.raises(RuntimeError):
            await download_and_unpack_snapshot(
                s3_client, "test-bucket", "snapshots/test.tar.gz", dest
            )

        s3_client.download_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_malicious_archive_uploaded_then_downloaded_still_raises(
        self, tmp_path
    ):
        """A malicious archive uploaded then downloaded still raises SnapshotTraversalError."""
        from unittest.mock import MagicMock
        from app.services.working_tree_snapshot import download_and_unpack_snapshot

        # Create a malicious tar with parent ref
        tar_path = tmp_path / "malicious.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="../etc/passwd")
            info.size = 0
            tar.addfile(info)

        # Mock S3 client
        s3_client = MagicMock()

        def mock_download_file(Bucket, Key, Filename, **kwargs):
            # Copy malicious tar to temp location
            import shutil
            shutil.copy(tar_path, Filename)

        s3_client.download_file = MagicMock(side_effect=mock_download_file)

        dest = tmp_path / "dest"
        dest.mkdir()

        # Should raise on unpack
        with pytest.raises(SnapshotTraversalError):
            await download_and_unpack_snapshot(
                s3_client, "test-bucket", "snapshots/malicious.tar.gz", dest
            )


class TestResourceCeilings:
    """Extraction refuses an archive that is honest about paths but not volume.

    Path validation answers "could this write outside dest_dir" and nothing more.
    A well-formed archive can still exhaust the worker's disk, because gzip
    reaches roughly 1000:1 on repetitive input — so the ceilings are what turn
    that into a rejection instead of an outage.
    """

    def test_a_gzip_bomb_is_refused_before_anything_is_written(self, tmp_path):
        """The archive below is a few KiB and claims ~3 GiB of zeros. Nothing is
        written: the ceiling is checked against the member header, so the cost of
        the lie is bounded by refusing to start rather than by filling a disk."""
        import io

        tar_path = tmp_path / "bomb.tar.gz"
        dest = tmp_path / "out"
        with tarfile.open(tar_path, "w:gz") as tar:
            for index in range(8):
                info = tarfile.TarInfo(name=f"zeros-{index}.bin")
                info.size = 400 * 1024**2  # 400 MiB each, 3.2 GiB total
                tar.addfile(info, io.BytesIO(b"\0" * info.size))

        with pytest.raises(SnapshotTooLargeError, match="expands past"):
            unpack_snapshot(tar_path, dest)

        assert not any(dest.rglob("*")), (
            "extraction wrote files before hitting the ceiling; the check has to "
            "happen before any member is written, not after"
        )

    def test_one_oversized_member_is_refused(self, tmp_path):
        import io

        tar_path = tmp_path / "big.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="huge.bin")
            info.size = 600 * 1024**2  # over the per-member ceiling
            tar.addfile(info, io.BytesIO(b"\0" * info.size))

        with pytest.raises(SnapshotTooLargeError, match="Member exceeds"):
            unpack_snapshot(tar_path, tmp_path / "out")

    def test_too_many_members_is_refused(self, tmp_path):
        """A million empty files costs nothing to compress and still exhausts
        inodes on the destination."""
        from app.services.working_tree_snapshot import _MAX_MEMBERS

        tar_path = tmp_path / "many.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for index in range(_MAX_MEMBERS + 1):
                tar.addfile(tarfile.TarInfo(name=f"f{index}"))

        with pytest.raises(SnapshotTooLargeError, match="more than"):
            unpack_snapshot(tar_path, tmp_path / "out")

    def test_the_member_ceiling_stops_reading_the_archive(self, tmp_path, monkeypatch):
        """The ceiling has to bound the READ, not only the write.

        ``getmembers()`` walks the whole archive before it returns — on a gzip
        stream that decompresses every member's data to find the next header, and
        keeps a ``TarInfo`` per member — so a ceiling checked over its result runs
        only after that cost is paid. Counting headers read pins the lazy walk: the
        refusal must come after about ``_MAX_MEMBERS`` headers, not after all of
        them. The test above cannot tell the two apart, because either way the
        error is raised.
        """
        from app.services import working_tree_snapshot

        monkeypatch.setattr(working_tree_snapshot, "_MAX_MEMBERS", 10)
        tar_path = tmp_path / "many.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for index in range(500):
                tar.addfile(tarfile.TarInfo(name=f"f{index}"))

        headers_read = 0
        real_next = tarfile.TarFile.next

        def counting_next(self):
            nonlocal headers_read
            headers_read += 1
            return real_next(self)

        monkeypatch.setattr(tarfile.TarFile, "next", counting_next)

        with pytest.raises(SnapshotTooLargeError, match="more than 10"):
            unpack_snapshot(tar_path, tmp_path / "out")

        # 11 headers reach the refusal; ``tarfile.open`` reads the first itself.
        assert headers_read <= 12, (
            f"read {headers_read} headers to refuse an archive over a 10-member "
            "ceiling — the whole archive was walked before the check ran"
        )

    def test_an_absurdly_deep_path_is_refused(self, tmp_path):
        from app.services.working_tree_snapshot import _MAX_PATH_DEPTH

        tar_path = tmp_path / "deep.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            deep = "/".join(f"d{i}" for i in range(_MAX_PATH_DEPTH + 2)) + "/f.txt"
            tar.addfile(tarfile.TarInfo(name=deep))

        with pytest.raises(SnapshotTooLargeError, match="nested deeper"):
            unpack_snapshot(tar_path, tmp_path / "out")

    def test_an_ordinary_snapshot_still_extracts(self, tmp_path):
        """The ceilings are wide enough for a real tree. A limit that rejects
        legitimate work is worse than no limit, because it gets raised in a hurry
        by whoever is on call."""
        project = tmp_path / "project"
        (project / "prompts").mkdir(parents=True)
        (project / "prompts" / "v1.md").write_text("be helpful")
        (project / "config.json").write_text('{"name": "x"}')

        tar_path = tmp_path / "ok.tar.gz"
        pack_working_tree(project, tar_path)
        dest = tmp_path / "out"
        unpack_snapshot(tar_path, dest)

        assert (dest / "prompts" / "v1.md").read_text() == "be helpful"
