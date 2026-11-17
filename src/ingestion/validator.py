"""PDF validation module for upload handling and security checks."""

from pathlib import Path
from typing import Tuple
import pymupdf

from ..config import MAX_FILE_SIZE_MB, sanitize_error


def validate_pdf(file_path: str | Path, max_size_mb: int = MAX_FILE_SIZE_MB) -> Tuple[bool, str]:
    """Validates an uploaded or catalogued PDF.

    Checks:
    - File existence and '.pdf' extension
    - File size <= max_size_mb
    - PDF magic bytes (%PDF-)
    - Non-encrypted and readable pages

    Returns:
        Tuple of (is_valid: bool, clean_message: str).
    """
    path = Path(file_path)
    file_display_name = path.name

    if not path.exists() or not path.is_file():
        return False, f"File '{file_display_name}' does not exist."

    if path.suffix.lower() != ".pdf":
        return False, f"Invalid format for '{file_display_name}': only .pdf files are accepted."

    # Size check
    try:
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File '{file_display_name}' ({file_size_mb:.1f} MB) exceeds maximum allowed size of {max_size_mb} MB."
    except Exception as e:
        return False, f"Unable to check size for '{file_display_name}': {type(e).__name__}"

    # Header check (PDF magic bytes)
    try:
        with open(path, "rb") as f:
            header = f.read(5)
            if not header.startswith(b"%PDF-"):
                return False, f"Invalid PDF header in '{file_display_name}': file is corrupted or not a valid PDF."
    except Exception as e:
        return False, f"Could not read header for '{file_display_name}': {type(e).__name__}"

    # Structural integrity and encryption check
    try:
        doc = pymupdf.open(path)
        if doc.is_encrypted:
            doc.close()
            return False, f"File '{file_display_name}' is password-protected or encrypted; skipping."
        if doc.page_count < 1:
            doc.close()
            return False, f"File '{file_display_name}' contains zero pages."
        doc.close()
    except Exception as e:
        # Never expose local paths or stack traces
        clean_err = sanitize_error(e)
        return False, f"Malformed PDF '{file_display_name}': {clean_err}"

    return True, ""
