"""Multimodal visual question answering, structured visual descriptions, and vision caching."""

import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from ..config import CACHE_DIR, DATA_DIR
from ..schemas import TextChunk, ChunkType
from .client import BaseLLMClient, MockProvider

logger = logging.getLogger("researchiq.vision")

VISUAL_DESCRIPTIONS_DIR = DATA_DIR / "visual_descriptions"
VISUAL_DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_VERSION = "v1.0"

VISUAL_DESCRIPTION_SYSTEM_PROMPT = """You are an expert scientific visual analysis system.
TASK: Analyze the provided image (diagram, chart, plot, architecture, flowchart, or table) and return a strictly structured JSON object.

SCHEMA:
{
  "figure_type": "architecture diagram | flowchart | bar/line/scatter chart | heatmap | screenshot | other",
  "summary": "High-level summary of what the visual depicts",
  "components": ["List of distinct visual boxes, components, or entities visible"],
  "text_labels_read": ["Exact text labels read off the image pixels"],
  "connections": ["Source -> Target (with arrow label if any)"],
  "data_flow": "Step-by-step description of data or processing flow if applicable",
  "chart": {
    "title": "Chart title if present",
    "x_axis": "X-axis label and scale",
    "y_axis": "Y-axis label and scale",
    "units": "Units of measurement (e.g. %, ms, score)",
    "legend": "Legend items and color mappings",
    "series": "Names of data series or models plotted",
    "trends": "Key visual trends (e.g., method A outperforms B as k increases)",
    "readable_values": ["Key specific data points or numbers read off axes/bars/markers"]
  },
  "unreadable_or_uncertain": ["List any text, labels, numbers, or arrows that are blurry, cut off, or illegible"],
  "confidence": "high | medium | low"
}

STRICT ANTI-HALLUCINATION RULES:
1. Describe ONLY what is directly visible in the image pixels.
2. If a label, value, number, or arrow is too small, faint, or unclear to read with certainty, list it under 'unreadable_or_uncertain' instead of guessing.
3. Do NOT copy claims from external text or captions as if they were visual observations.
4. NEVER invent or extrapolate numerical values. If values cannot be read precisely from axis markings, note this in 'unreadable_or_uncertain'.
5. If the image resolution is low or labels are degraded, set confidence to 'low'.
6. For multi-panel figures (e.g., (a), (b), (c)), explicitly describe each panel in components and summary.
7. Return valid JSON only, without markdown code fences or conversational prose.
"""

VISUAL_QA_SYSTEM_PROMPT = """You are an expert scientific multimodal vision analyst for Advanced Retrieval-Augmented Generation papers.

TASK:
Answer the user's question about the figure, chart, or diagram.

REQUIRED ANSWER FORMAT:
You MUST structure your response into three clearly separated sections:
1. [PAPER CAPTION & TEXT STATEMENT]: What the paper's caption and adjacent text explicitly state.
2. [DIRECT VISUAL OBSERVATION]: What is directly observed from the image pixels (modules, arrows, curves, axes, trends, markers).
3. [AI-GENERATED DESCRIPTION DETAILS]: Any relevant details from the pre-generated visual description (explicitly labeled as AI-generated).

STRICT UNCERTAINTY RULES:
- If a requested detail, component, arrow, or number is NOT clearly visible or illegible in the image, you MUST explicitly state: "Cannot be determined from the image: [detail] is not visible."
- Never fabricate or guess unobserved information.
- All claims must cite the parent figure in the standard citation format: [{paper_id}, p.{page}, {figure_number}].
"""


def compute_image_hash(image_path: Path) -> str:
    """Computes SHA-256 hash of image file content."""
    h = hashlib.sha256()
    if image_path.exists() and image_path.is_file():
        try:
            with open(image_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return hashlib.sha256(str(image_path).encode()).hexdigest()
    return hashlib.sha256(str(image_path).encode()).hexdigest()


def generate_visual_description(
    chunk: TextChunk,
    client: BaseLLMClient,
    force_refresh: bool = False,
    prompt_version: str = PROMPT_VERSION,
) -> Dict[str, Any]:
    """Generates or retrieves structured visual description with caching and anti-hallucination rules."""
    img_p = Path(chunk.image_path) if chunk.image_path else Path("")
    img_hash = compute_image_hash(img_p) if img_p.exists() else "no_image"

    paper_desc_dir = VISUAL_DESCRIPTIONS_DIR / chunk.paper_id
    paper_desc_dir.mkdir(parents=True, exist_ok=True)
    figure_desc_file = paper_desc_dir / f"{chunk.chunk_id}.json"

    cache_file = CACHE_DIR / f"visdesc_{img_hash}_{prompt_version}.json"

    # 1. Check persistent storage and cache
    if not force_refresh:
        if figure_desc_file.exists():
            try:
                with open(figure_desc_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    with open(figure_desc_file, "w", encoding="utf-8") as df:
                        json.dump(data, df, indent=2)
                    return data
            except Exception:
                pass

    # 2. Check if MockProvider without canned response
    if isinstance(client, MockProvider) and client.canned_response is None:
        mock_data = _generate_mock_visual_description(chunk, img_hash, prompt_version)
        with open(figure_desc_file, "w", encoding="utf-8") as f:
            json.dump(mock_data, f, indent=2)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(mock_data, f, indent=2)
        return mock_data

    # 3. Call Vision Model
    prompt = (
        f"Analyze this scientific figure from paper {chunk.paper_id} (Page {chunk.page}, {chunk.element_number or 'Figure'}).\n"
        f"Extracted Caption (for reference only, do not copy as visual observation): {chunk.caption or 'None'}\n\n"
        f"Return the strictly structured JSON visual description complying with all anti-hallucination rules."
    )

    raw_resp = client.generate(
        prompt=prompt,
        system=VISUAL_DESCRIPTION_SYSTEM_PROMPT,
        images=[img_p] if img_p.exists() else None,
    )

    # 4. Parse JSON robustly
    parsed = _parse_json_robustly(raw_resp)

    # 5. Attach provenance fields
    model_name = getattr(client, "model_name", "gemini-2.5-flash")
    provenance = {
        "source": "ai_generated_gemini_vision",
        "model_name": model_name,
        "prompt_version": prompt_version,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "parent_chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "page": chunk.page,
        "figure_number": chunk.element_number or f"Figure {chunk.page}",
        "image_path": str(img_p),
        "image_hash": img_hash,
    }

    result = {**provenance, "description": parsed}

    # Save to disk
    with open(figure_desc_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def _parse_json_robustly(text: str) -> Dict[str, Any]:
    """Strips markdown code fences and parses JSON safely."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()

    try:
        return json.loads(t)
    except Exception:
        # Fallback dictionary for malformed model JSON
        return {
            "figure_type": "other",
            "summary": t[:300],
            "components": [],
            "text_labels_read": [],
            "connections": [],
            "data_flow": "",
            "chart": {},
            "unreadable_or_uncertain": ["Output returned unformatted text."],
            "confidence": "low",
        }


def _generate_mock_visual_description(
    chunk: TextChunk, img_hash: str, prompt_version: str
) -> Dict[str, Any]:
    """Generates deterministic mock visual description for offline tests."""
    is_chart = any(w in (chunk.caption or "").lower() for w in ["curve", "plot", "recall", "score", "f1", "ablation", "heatmap"])
    ftype = "bar/line/scatter chart" if is_chart else "architecture diagram"

    desc = {
        "figure_type": ftype,
        "summary": f"Visual representation of {chunk.caption or 'scientific concept'} on page {chunk.page}.",
        "components": ["Input Query", "Retriever Module", "Reranker", "Generator"],
        "text_labels_read": ["Query", "BM25", "Dense Encoder", "RRF Score", "Output Answer"],
        "connections": ["Input Query -> Retriever Module", "Retriever Module -> Reranker", "Reranker -> Generator"],
        "data_flow": "Query is routed to dense and sparse retrievers, fused via RRF, and passed to LLM.",
        "chart": {
            "title": chunk.caption or "Metric Comparison",
            "x_axis": "k (Top-k candidates)",
            "y_axis": "Recall@k",
            "units": "%",
            "legend": "BM25, Dense, Hybrid",
            "series": "BM25 vs Dense vs Hybrid",
            "trends": "Hybrid consistently achieves highest recall across all k values.",
            "readable_values": ["k=5: 64.2%", "k=10: 78.5%", "k=20: 86.1%"],
        } if is_chart else {},
        "unreadable_or_uncertain": ["Sub-label font below 6pt unreadable"],
        "confidence": "high",
    }

    return {
        "source": "ai_generated_gemini_vision",
        "model_name": "mock-vision-provider",
        "prompt_version": prompt_version,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "parent_chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "page": chunk.page,
        "figure_number": chunk.element_number or f"Figure {chunk.page}",
        "image_path": chunk.image_path or "",
        "image_hash": img_hash,
        "description": desc,
    }


def build_visual_description_chunk(fig_chunk: TextChunk, desc_record: Dict[str, Any]) -> TextChunk:
    """Builds a new visual_description TextChunk linked to its parent figure."""
    d = desc_record.get("description", desc_record)
    summary = d.get("summary", "")
    components = ", ".join(d.get("components", []))
    labels = ", ".join(d.get("text_labels_read", []))
    connections = " | ".join(d.get("connections", []))
    flow = d.get("data_flow", "")
    chart_info = d.get("chart", {})
    chart_str = ""
    if chart_info:
        chart_str = (
            f"Chart Title: {chart_info.get('title', '')}\n"
            f"Axes: X={chart_info.get('x_axis', '')}, Y={chart_info.get('y_axis', '')} ({chart_info.get('units', '')})\n"
            f"Trends: {chart_info.get('trends', '')}\n"
            f"Values: {', '.join(chart_info.get('readable_values', []))}\n"
        )
    uncertain = ", ".join(d.get("unreadable_or_uncertain", []))

    desc_text = (
        f"[AI-GENERATED VISUAL DESCRIPTION for {fig_chunk.element_number or 'Figure'} in {fig_chunk.paper_id} (p.{fig_chunk.page})]\n"
        f"Figure Type: {d.get('figure_type', 'unknown')}\n"
        f"Summary: {summary}\n"
        f"Visible Components: {components}\n"
        f"Readable Labels: {labels}\n"
        f"Connections & Flow: {connections}. {flow}\n"
        f"{chart_str}"
        f"Uncertain/Unreadable: {uncertain}\n"
        f"Confidence: {d.get('confidence', 'medium')}"
    )

    return TextChunk(
        chunk_id=f"{fig_chunk.chunk_id}_visdesc",
        parent_chunk_id=fig_chunk.chunk_id,
        paper_id=fig_chunk.paper_id,
        paper_title=fig_chunk.paper_title,
        page=fig_chunk.page,
        section=f"Visual Description: {fig_chunk.element_number or 'Figure'}",
        text=desc_text,
        chunk_type=ChunkType.VISUAL_DESCRIPTION.value,
        element_number=fig_chunk.element_number,
        figure_number=fig_chunk.element_number,
        image_path=fig_chunk.image_path,
        bbox=fig_chunk.bbox,
        token_count=len(desc_text.split()),
        visual_description=d,
        metadata={
            "source": desc_record.get("source", "ai_generated_gemini_vision"),
            "model_name": desc_record.get("model_name", "gemini-2.5-flash"),
            "prompt_version": desc_record.get("prompt_version", PROMPT_VERSION),
            "parent_chunk_id": fig_chunk.chunk_id,
            "image_hash": desc_record.get("image_hash", ""),
        },
    )


def answer_visual_question(
    question: str,
    parent_chunk: TextChunk,
    description_data: Optional[Dict[str, Any]],
    client: BaseLLMClient,
    force_refresh: bool = False,
) -> Tuple[str, bool]:
    """Answers question about a visual element separating caption, direct observation, and AI description."""
    img_p = Path(parent_chunk.image_path) if parent_chunk.image_path else Path("")
    cache_key = hashlib.sha256(f"{img_p}_{question}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"visqa_{cache_key}.json"

    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return cached.get("answer", ""), True
        except Exception:
            pass

    desc_details = ""
    if description_data:
        d = description_data.get("description", description_data)
        desc_details = json.dumps(d, indent=2)

    prompt = (
        f"USER QUESTION: {question}\n\n"
        f"=== CONTEXT SOURCE 1: PAPER CAPTION ===\n"
        f"{parent_chunk.caption or 'No caption available'}\n\n"
        f"=== CONTEXT SOURCE 2: NEARBY PAPER BODY TEXT ===\n"
        f"{parent_chunk.nearby_text or 'No nearby text available'}\n\n"
        f"=== CONTEXT SOURCE 3: AI-GENERATED VISUAL DESCRIPTION ===\n"
        f"{desc_details or 'None'}\n\n"
        f"Answer the user question adhering strictly to the three-part required format and uncertainty rules."
    )

    try:
        raw_answer = client.generate(
            prompt=prompt,
            system=VISUAL_QA_SYSTEM_PROMPT,
            images=[img_p] if img_p.exists() else None,
        )
    except Exception as e:
        raw_answer = (
            f"1. [PAPER CAPTION & TEXT STATEMENT]: {parent_chunk.caption or 'None'}\n"
            f"2. [DIRECT VISUAL OBSERVATION]: Direct vision model evaluation failed: {type(e).__name__}.\n"
            f"3. [AI-GENERATED DESCRIPTION DETAILS]: Pre-indexed description indicates components {desc_details[:150]}..."
        )

    # Save to disk cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"answer": raw_answer, "question": question, "image": str(img_p)}, f, indent=2)
    except Exception:
        pass

    return raw_answer, False


def analyze_visual_chunk(
    chunk: TextChunk,
    question: str,
    client: BaseLLMClient,
) -> Tuple[str, bool]:
    """Backwards-compatible helper for visual chunk analysis."""
    desc = getattr(chunk, "visual_description", None)
    return answer_visual_question(question, chunk, desc, client)
    img_p = Path(parent_chunk.image_path) if parent_chunk.image_path else Path("")
    cache_key = hashlib.sha256(f"{img_p}_{question}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"visqa_{cache_key}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return cached.get("answer", ""), True
        except Exception:
            pass

    desc_details = ""
    if description_data:
        d = description_data.get("description", description_data)
        desc_details = json.dumps(d, indent=2)

    prompt = (
        f"USER QUESTION: {question}\n\n"
        f"=== CONTEXT SOURCE 1: PAPER CAPTION ===\n"
        f"{parent_chunk.caption or 'No caption available'}\n\n"
        f"=== CONTEXT SOURCE 2: NEARBY PAPER BODY TEXT ===\n"
        f"{parent_chunk.nearby_text or 'No nearby text available'}\n\n"
        f"=== CONTEXT SOURCE 3: AI-GENERATED VISUAL DESCRIPTION ===\n"
        f"{desc_details or 'None'}\n\n"
        f"Answer the user question adhering strictly to the three-part required format and uncertainty rules."
    )

    try:
        raw_answer = client.generate(
            prompt=prompt,
            system=VISUAL_QA_SYSTEM_PROMPT,
            images=[img_p] if img_p.exists() else None,
        )
    except Exception as e:
        raw_answer = (
            f"1. [PAPER CAPTION & TEXT STATEMENT]: {parent_chunk.caption or 'None'}\n"
            f"2. [DIRECT VISUAL OBSERVATION]: Direct vision model evaluation failed: {type(e).__name__}.\n"
            f"3. [AI-GENERATED DESCRIPTION DETAILS]: Pre-indexed description indicates components {desc_details[:150]}..."
        )

    # Save to disk cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"answer": raw_answer, "question": question, "image": str(img_p)}, f, indent=2)
    except Exception:
        pass

    return raw_answer, False
