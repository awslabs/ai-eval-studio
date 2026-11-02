# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The coordinator's TLS entrypoint produces a usable certificate and hands it to uvicorn."""

import ssl
import stat
from unittest.mock import patch

from app.scripts import serve_tls


def test_generated_pair_loads_as_a_server_certificate(tmp_path):
    """The key and certificate load into a TLS server context, as uvicorn loads them."""
    key_path, cert_path = serve_tls.write_self_signed_certificate(tmp_path)

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))


def test_private_key_is_readable_only_by_its_owner(tmp_path):
    """The key file is created 0600, so no other user can read it."""
    key_path, _ = serve_tls.write_self_signed_certificate(tmp_path)

    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600


def test_main_passes_arguments_through_and_adds_the_certificate(monkeypatch, tmp_path):
    """uvicorn receives every argument given, plus the generated key and certificate."""
    monkeypatch.setattr(serve_tls.tempfile, "mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["serve_tls", "app.main:app", "--port", "8100", "--ws-max-size", "65536"],
    )
    with patch("uvicorn.main") as uvicorn_main:
        serve_tls.main()

    args = uvicorn_main.call_args.kwargs["args"]
    assert args[:5] == ["app.main:app", "--port", "8100", "--ws-max-size", "65536"]
    key_file = args[args.index("--ssl-keyfile") + 1]
    cert_file = args[args.index("--ssl-certfile") + 1]
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
