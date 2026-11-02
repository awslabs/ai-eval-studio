# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the multi-file upload endpoint.

Covers happy-path uploads (single, multi, zip), size caps, OS-detritus
filtering, zip-stem extraction naming, sibling-collision handling, and
the security guards (zip-slip, zip-bomb, symlinks, bad filenames).
Also covers the presigned-PUT flow for large uploads (init + complete),
with boto3 stubbed via ``unittest.mock``.
"""
from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app


@pytest.fixture
def app(tmp_path):
    import app.main as main_module
    original = main_module.PROJECTS_DIR
    original_pm = main_module._project_manager
    main_module.PROJECTS_DIR = tmp_path / "projects"
    main_module.PROJECTS_DIR.mkdir()
    main_module._project_manager = None
    application = create_app()
    yield application
    main_module.PROJECTS_DIR = original
    main_module._project_manager = original_pm


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _project_dir(app, project_id: str):
    """Return the filesystem path of a project created in this app."""
    import app.main as main_module
    return main_module.PROJECTS_DIR / project_id


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    """Build a zip in-memory from ``(name, content)`` tuples."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries:
            zf.writestr(name, content)
    return buf.getvalue()


async def _create_project(client) -> str:
    resp = await client.post("/api/projects", json={"name": "Upload Test"})
    return resp.json()["id"]


class TestSingleFileUpload:
    async def test_loose_file_lands_in_data_uploads(self, client, app):
        project_id = await _create_project(client)
        csv = b"id,input,expected\n1,hello,world\n"
        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("cases.csv", io.BytesIO(csv), "text/csv"))],
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["files"] == ["data/uploads/cases.csv"]
        assert data["total_bytes"] == len(csv)

        on_disk = _project_dir(app, project_id) / "data" / "uploads" / "cases.csv"
        assert on_disk.read_bytes() == csv

    async def test_project_not_found(self, client):
        # A valid-format UUID that doesn't correspond to any project. The
        # auth dependency rejects malformed (non-UUID) values with 400
        # *before* the existence check, so a real UUID is needed to land on
        # the 404 path.
        resp = await client.post(
            "/api/projects/00000000-0000-0000-0000-000000000000/upload",
            files=[("files", ("x.csv", io.BytesIO(b"a,b"), "text/csv"))],
        )
        assert resp.status_code == 404

    async def test_invalid_filename_rejected(self, client):
        project_id = await _create_project(client)
        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("..", io.BytesIO(b"x"), "text/plain"))],
        )
        assert resp.status_code == 400


class TestMultiFileUpload:
    async def test_two_loose_files_one_request(self, client, app):
        project_id = await _create_project(client)
        csv = b"a,b\n1,2\n"
        md = b"# Notes\n"
        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[
                ("files", ("cases.csv", io.BytesIO(csv), "text/csv")),
                ("files", ("readme.md", io.BytesIO(md), "text/markdown")),
            ],
        )
        assert resp.status_code == 200, resp.text
        files = resp.json()["files"]
        assert set(files) == {"data/uploads/cases.csv", "data/uploads/readme.md"}

    async def test_duplicate_loose_filename_gets_suffix(self, client, app):
        project_id = await _create_project(client)
        await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("a.csv", io.BytesIO(b"first"), "text/csv"))],
        )
        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("a.csv", io.BytesIO(b"second"), "text/csv"))],
        )
        assert resp.status_code == 200
        assert resp.json()["files"] == ["data/uploads/a_2.csv"]
        root = _project_dir(app, project_id) / "data" / "uploads"
        assert (root / "a.csv").read_bytes() == b"first"
        assert (root / "a_2.csv").read_bytes() == b"second"


class TestZipUpload:
    async def test_zip_preserved_and_extracted_into_stem_dir(self, client, app):
        project_id = await _create_project(client)
        raw = _zip_bytes([
            ("cases.csv", b"a,b\n1,2\n"),
            ("docs/manual.md", b"# Manual"),
        ])
        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("dataset.zip", io.BytesIO(raw), "application/zip"))],
        )
        assert resp.status_code == 200, resp.text

        files = resp.json()["files"]
        assert "data/uploads/dataset.zip" in files
        assert "data/uploads/dataset/cases.csv" in files
        assert "data/uploads/dataset/docs/manual.md" in files

        root = _project_dir(app, project_id) / "data" / "uploads"
        assert (root / "dataset.zip").read_bytes() == raw
        assert (root / "dataset" / "cases.csv").is_file()
        assert (root / "dataset" / "docs" / "manual.md").is_file()

    async def test_zip_collision_gets_suffix(self, client, app):
        project_id = await _create_project(client)
        raw = _zip_bytes([("cases.csv", b"hi\n")])

        r1 = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("kit.zip", io.BytesIO(raw), "application/zip"))],
        )
        r2 = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("kit.zip", io.BytesIO(raw), "application/zip"))],
        )
        assert r1.status_code == 200 and r2.status_code == 200

        root = _project_dir(app, project_id) / "data" / "uploads"
        assert (root / "kit.zip").is_file()
        assert (root / "kit_2.zip").is_file()
        assert (root / "kit" / "cases.csv").is_file()
        assert (root / "kit_2" / "cases.csv").is_file()

    async def test_os_detritus_filtered(self, client, app):
        project_id = await _create_project(client)
        raw = _zip_bytes([
            ("real.csv", b"x\n"),
            ("__MACOSX/real.csv", b"junk"),
            ("__MACOSX/._real.csv", b"junk"),
            (".DS_Store", b"junk"),
            ("subdir/.DS_Store", b"junk"),
            ("subdir/._hidden", b"junk"),
            ("subdir/payload.md", b"ok"),
        ])
        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("pack.zip", io.BytesIO(raw), "application/zip"))],
        )
        assert resp.status_code == 200, resp.text

        root = _project_dir(app, project_id) / "data" / "uploads" / "pack"
        present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        assert "real.csv" in present
        assert "subdir/payload.md" in present
        # None of the filtered entries should have been written.
        assert not any(name.startswith("__MACOSX") for name in present)
        assert ".DS_Store" not in present
        assert "subdir/.DS_Store" not in present
        assert "subdir/._hidden" not in present

    async def test_zip_slip_rejected(self, client):
        project_id = await _create_project(client)
        content = _zip_bytes([("../../../etc/evil", b"pwn")])
        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("bad.zip", io.BytesIO(content), "application/zip"))],
        )
        assert resp.status_code == 400
        assert "escapes" in resp.text.lower()

    async def test_zip_bomb_rejected(self, client):
        project_id = await _create_project(client)
        big = b"A" * (550 * 1024 * 1024)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("bomb.txt", big)
        content = buf.getvalue()
        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("bomb.zip", io.BytesIO(content), "application/zip"))],
        )
        assert resp.status_code == 413

    async def test_malformed_zip(self, client):
        project_id = await _create_project(client)
        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("broken.zip", io.BytesIO(b"not a zip"), "application/zip"))],
        )
        assert resp.status_code == 400

    async def test_symlink_entries_skipped(self, client, app):
        """Symlinks inside uploaded zips must not materialize on disk.

        Covers two variants of the attack: (a) explicit S_IFLNK mode bits
        (0o120000 in the high 16 bits of ``external_attr``), and (b) the
        crafted ``external_attr=0`` that an earlier implementation would
        accept because a truthiness guard bypassed the mode check.
        """
        project_id = await _create_project(client)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            # Regular file — should be written.
            zf.writestr("keep.csv", b"ok\n")

            # (a) Explicit symlink: S_IFLNK in the S_IFMT bits.
            symlink_info = zipfile.ZipInfo("evil_link")
            symlink_info.external_attr = 0o120777 << 16
            zf.writestr(symlink_info, b"/etc/passwd")

            # (b) Zeroed external_attr symlink payload. zipfile can't
            # label this as S_IFLNK without setting mode bits, so we
            # simulate the "non-regular, non-dir type" case by writing
            # a block-device mode (0o060000). A crafted archive could
            # use any non-regular S_IFMT here.
            device_info = zipfile.ZipInfo("evil_device")
            device_info.external_attr = 0o060777 << 16
            zf.writestr(device_info, b"should not land")
        content = buf.getvalue()

        resp = await client.post(
            f"/api/projects/{project_id}/upload",
            files=[("files", ("mixed.zip", io.BytesIO(content), "application/zip"))],
        )
        assert resp.status_code == 200, resp.text

        root = _project_dir(app, project_id) / "data" / "uploads" / "mixed"
        present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        assert "keep.csv" in present
        assert "evil_link" not in present
        assert "evil_device" not in present


class TestPresignedUploadFlow:
    """Large uploads via direct-to-S3 presigned PUT.

    boto3 is stubbed at the module level; we don't make real AWS calls.
    The contract we assert: init returns a URL + upload_id, complete
    streams the staged bytes into the project's data/uploads dir.
    """

    async def test_init_503_when_staging_unconfigured(self, client, monkeypatch):
        """Without ``S3_UPLOADS_STAGING_BUCKET`` the endpoint refuses."""
        monkeypatch.delenv("S3_UPLOADS_STAGING_BUCKET", raising=False)
        project_id = await _create_project(client)
        resp = await client.post(
            f"/api/projects/{project_id}/uploads/init",
            json={"filename": "huge.csv", "size_bytes": 10 * 1024 * 1024 * 1024},
        )
        assert resp.status_code == 503

    async def test_init_413_for_oversize_declaration(self, client, monkeypatch):
        """Caller-declared size above the cap is rejected before signing."""
        monkeypatch.setenv("S3_UPLOADS_STAGING_BUCKET", "test-staging")
        project_id = await _create_project(client)

        with patch("boto3.client") as mock_client:
            mock_client.return_value = MagicMock()
            resp = await client.post(
                f"/api/projects/{project_id}/uploads/init",
                json={
                    "filename": "huge.csv",
                    # 10 GB > 5 GB cap.
                    "size_bytes": 10 * 1024 * 1024 * 1024,
                },
            )
        assert resp.status_code == 413

    async def test_init_400_on_bad_filename(self, client, monkeypatch):
        monkeypatch.setenv("S3_UPLOADS_STAGING_BUCKET", "test-staging")
        project_id = await _create_project(client)
        resp = await client.post(
            f"/api/projects/{project_id}/uploads/init",
            json={"filename": "..", "size_bytes": 1024},
        )
        assert resp.status_code == 400

    async def test_init_returns_presigned_url(self, client, monkeypatch):
        monkeypatch.setenv("S3_UPLOADS_STAGING_BUCKET", "test-staging")
        project_id = await _create_project(client)

        fake_s3 = MagicMock()
        fake_s3.generate_presigned_url.return_value = (
            "https://test-staging.s3.amazonaws.com/key?X-Amz-Signature=abc"
        )
        with patch("boto3.client", return_value=fake_s3):
            resp = await client.post(
                f"/api/projects/{project_id}/uploads/init",
                json={
                    "filename": "data.csv",
                    "content_type": "text/csv",
                    "size_bytes": 12345,
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["presigned_url"].startswith("https://")
        assert "upload_id" in body
        assert body["target_key"].startswith(f"users/")
        assert body["target_key"].endswith("/data.csv")
        # Presign was called with the expected bucket + key shape.
        call = fake_s3.generate_presigned_url.call_args
        assert call.args[0] == "put_object"
        params = call.kwargs["Params"]
        assert params["Bucket"] == "test-staging"
        assert params["ContentType"] == "text/csv"
        # Content-Length must be signed in so S3 rejects PUTs whose
        # body length doesn't match — see the upload.py docstring.
        assert params["ContentLength"] == 12345

    async def test_complete_copies_s3_to_efs(self, client, app, monkeypatch):
        """End-to-end: init → fake S3 PUT → complete → file on disk."""
        monkeypatch.setenv("S3_UPLOADS_STAGING_BUCKET", "test-staging")
        project_id = await _create_project(client)

        fake_s3 = MagicMock()
        fake_s3.generate_presigned_url.return_value = (
            "https://test-staging.s3.amazonaws.com/x?sig=y"
        )
        # head_object returns the staged size; get_object returns a
        # streaming body that yields the payload in one chunk.
        payload = b"id,input,expected\n1,a,b\n2,c,d\n"
        fake_s3.head_object.return_value = {"ContentLength": len(payload)}

        body_stream = MagicMock()
        # boto3's StreamingBody.read(n) returns up to n bytes; we simulate
        # the full payload then EOF. The endpoint reads in 8 MiB chunks
        # so two read() calls suffice.
        body_stream.read.side_effect = [payload, b""]
        fake_s3.get_object.return_value = {"Body": body_stream}

        with patch("boto3.client", return_value=fake_s3):
            init_resp = await client.post(
                f"/api/projects/{project_id}/uploads/init",
                json={"filename": "cases.csv", "size_bytes": len(payload)},
            )
            assert init_resp.status_code == 200, init_resp.text
            upload_id = init_resp.json()["upload_id"]

            complete_resp = await client.post(
                f"/api/projects/{project_id}/uploads/complete",
                json={"upload_id": upload_id, "filename": "cases.csv"},
            )

        assert complete_resp.status_code == 200, complete_resp.text
        body = complete_resp.json()
        assert body["files"] == ["data/uploads/cases.csv"]
        assert body["total_bytes"] == len(payload)

        on_disk = _project_dir(app, project_id) / "data" / "uploads" / "cases.csv"
        assert on_disk.read_bytes() == payload

        # Staging delete was attempted.
        fake_s3.delete_object.assert_called_once()

    async def test_complete_404_when_staged_missing(self, client, monkeypatch):
        """If the browser never PUT to S3, complete returns 404."""
        monkeypatch.setenv("S3_UPLOADS_STAGING_BUCKET", "test-staging")
        project_id = await _create_project(client)

        from botocore.exceptions import ClientError

        fake_s3 = MagicMock()
        fake_s3.head_object.side_effect = ClientError(
            error_response={"Error": {"Code": "404", "Message": "Not Found"}},
            operation_name="HeadObject",
        )

        with patch("boto3.client", return_value=fake_s3):
            resp = await client.post(
                f"/api/projects/{project_id}/uploads/complete",
                json={
                    "upload_id": "00000000-0000-0000-0000-000000000000",
                    "filename": "missing.csv",
                },
            )
        assert resp.status_code == 404
