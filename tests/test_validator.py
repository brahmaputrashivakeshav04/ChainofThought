"""Unit tests for PDF validator."""

from pathlib import Path
from src.ingestion.validator import validate_pdf


def test_validator_nonexistent_file(tmp_path: Path):
    valid, msg = validate_pdf(tmp_path / "ghost.pdf")
    assert not valid
    assert "does not exist" in msg.lower()


def test_validator_invalid_header(tmp_path: Path):
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("This is not a real PDF document.")
    valid, msg = validate_pdf(fake_pdf)
    assert not valid
    assert "invalid pdf header" in msg.lower()


def test_validator_oversized(tmp_path: Path):
    fake_large = tmp_path / "large.pdf"
    fake_large.write_bytes(b"%PDF-1.4\n" + b"0" * (1024 * 1024 * 2))
    valid, msg = validate_pdf(fake_large, max_size_mb=1)
    assert not valid
    assert "exceeds maximum allowed size" in msg.lower()
