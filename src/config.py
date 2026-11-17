"""ResearchIQ configuration module.

Handles environment variables, default directory paths, model hyperparameters,
and security constraints. No hard-coded credentials exist here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
RESEARCH_DATA_DIR = BASE_DIR / "research-data"
FIGURES_DIR = DATA_DIR / "figures"
TABLES_DIR = DATA_DIR / "tables"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
CATALOGUE_PATH = BASE_DIR / "papers_catalogue.csv"

# Ensure directories exist
for path in (PAPERS_DIR, FIGURES_DIR, TABLES_DIR, INDEX_DIR, CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)

# Security & Upload Validation
MAX_FILE_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "30"))
ALLOWED_MIME_TYPES = ["application/pdf"]
ALLOW_INSECURE_EVAL = os.getenv("ALLOW_INSECURE_EVAL", "false").lower() == "true"

# Provider Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Fallback OpenRouter/OpenAI-compatible Provider (optional)
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "https://openrouter.ai/api/v1")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "meta-llama/llama-3-8b-instruct")

# Model Identifiers
DEFAULT_LLM_MODEL = GEMINI_MODEL
DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", GEMINI_MODEL)
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval & Hybrid Search Hyperparameters
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "6"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.005"))
CHUNK_SIZE = 500  # Token / word approx target
CHUNK_OVERLAP = 100

# Error Masking Helper
def sanitize_error(err: Exception) -> str:
    """Mask stack traces, internal paths, and secrets before displaying to users."""
    msg = str(err)
    # Strip local path prefixes
    base_str = str(BASE_DIR)
    if base_str in msg:
        msg = msg.replace(base_str, "[WORKSPACE]")
    # Never expose API key fragments
    for key in (GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY):
        if key and len(key) > 5 and key in msg:
            msg = msg.replace(key, "sk-***")
    return f"Operation failed: {msg.splitlines()[-1] if msg.splitlines() else 'Unknown error'}"
