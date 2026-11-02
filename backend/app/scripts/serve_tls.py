# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinator entrypoint that serves uvicorn over TLS with a per-task certificate.

Invoked as the coordinator container command::

    python -m app.scripts.serve_tls app.main:app --host 0.0.0.0 --port 8100 ...

Every argument after the module name is passed to uvicorn unchanged, with
``--ssl-keyfile`` and ``--ssl-certfile`` appended.

WHY a self-signed certificate: the hop it protects is the internal ALB to this
task. ALB encrypts that hop but does not validate the target's certificate, so
a CA-issued certificate would add nothing and would need distributing. The key
is generated in the task at start-up, written only to a private temporary
directory, and never leaves the task.

Local development does not use this module: ``scripts/start.sh`` runs plain
uvicorn over HTTP on localhost.
"""

from __future__ import annotations

import datetime
import os
import socket
import sys
import tempfile
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

# Long enough that a task never outlives its certificate, since the certificate
# is never renewed in place. Fargate tasks are replaced on every deploy.
CERTIFICATE_VALIDITY = datetime.timedelta(days=365)


def write_self_signed_certificate(directory: Path) -> tuple[Path, Path]:
    """Generate a key and a self-signed certificate under ``directory``.

    Args:
        directory: An existing directory only this process's user can read.

    Returns:
        The key file path and the certificate file path, in that order.
    """
    key = ec.generate_private_key(ec.SECP256R1())
    hostname = socket.gethostname()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + CERTIFICATE_VALIDITY)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False
        )
        .sign(key, hashes.SHA256())
    )

    key_path = directory / "key.pem"
    cert_path = directory / "cert.pem"
    # Created with mode 0600 so the key is never readable by another user, not
    # even between the write and a later chmod.
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as key_file:
        key_file.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return key_path, cert_path


def main() -> None:
    """Generate the certificate, then run uvicorn with it."""
    import uvicorn

    # mkdtemp creates the directory with mode 0700. The image's source tree is
    # read-only, so the certificate has to live outside it.
    directory = Path(tempfile.mkdtemp(prefix="coordinator-tls-"))
    key_path, cert_path = write_self_signed_certificate(directory)
    uvicorn.main(
        args=[
            *sys.argv[1:],
            "--ssl-keyfile",
            str(key_path),
            "--ssl-certfile",
            str(cert_path),
        ],
        prog_name="uvicorn",
    )


if __name__ == "__main__":
    main()
