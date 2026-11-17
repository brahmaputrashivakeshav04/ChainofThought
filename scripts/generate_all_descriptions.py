"""Visual descriptions generation, indexing, and dry run script."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import INDEX_DIR, FIGURES_DIR
from src.llm.client import get_llm_client, MockProvider
from src.llm.multimodal_qa import (
    generate_visual_description,
    build_visual_description_chunk,
    VISUAL_DESCRIPTIONS_DIR,
)
from src.retrieval.vector_store import VectorIndex
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import EmbeddingProvider

def run_visual_pipeline():
    print("==================================================")
    print("STEP 2: VISUAL DESCRIPTIONS & STEP 3: INDEXING")
    print("==================================================")

    # 1. Load existing index
    vec_index = VectorIndex()
    vec_index.load(INDEX_DIR)
    bm25_idx = BM25Index()
    bm25_idx.load(INDEX_DIR)
    embedder = EmbeddingProvider()

    before_count = len(vec_index.chunks)
    print(f"Chunks before visual descriptions: {before_count}")

    # Extract all figure chunks
    figure_chunks = [c for c in vec_index.chunks if c.chunk_type == "figure"]
    print(f"Found {len(figure_chunks)} figure chunks to analyze.")

    # Select 1 architecture diagram and 1 chart for DRY RUN
    arch_chunk = None
    chart_chunk = None
    for fc in figure_chunks:
        cap_l = (fc.caption or "").lower()
        if not arch_chunk and any(w in cap_l for w in ["architecture", "overview", "framework", "pipeline"]):
            arch_chunk = fc
        if not chart_chunk and any(w in cap_l for w in ["curve", "plot", "recall", "score", "f1", "ablation", "heatmap"]):
            chart_chunk = fc

    if not arch_chunk and figure_chunks:
        arch_chunk = figure_chunks[0]
    if not chart_chunk and len(figure_chunks) > 1:
        chart_chunk = figure_chunks[1]

    client = get_llm_client()
    print(f"Using LLM/Vision Client: {type(client).__name__}")

    # DRY RUN 1: Architecture diagram
    print("\n--- DRY RUN: Architecture Diagram ---")
    print(f"Figure: {arch_chunk.chunk_id} ({arch_chunk.paper_id} p.{arch_chunk.page} - {arch_chunk.element_number})")
    print(f"Caption: {arch_chunk.caption[:80]}")
    arch_desc = generate_visual_description(arch_chunk, client=client, force_refresh=True)
    print("Full JSON Visual Description:")
    print(json.dumps(arch_desc, indent=2))

    # DRY RUN 2: Chart
    print("\n--- DRY RUN: Chart / Plot ---")
    print(f"Figure: {chart_chunk.chunk_id} ({chart_chunk.paper_id} p.{chart_chunk.page} - {chart_chunk.element_number})")
    print(f"Caption: {chart_chunk.caption[:80]}")
    chart_desc = generate_visual_description(chart_chunk, client=client, force_refresh=True)
    print("Full JSON Visual Description:")
    print(json.dumps(chart_desc, indent=2))

    # Process all figures sequentially with caching
    print("\n--- Generating Visual Descriptions for All Figures ---")
    new_desc_chunks = []
    for idx, fc in enumerate(figure_chunks):
        print(f"[{idx+1}/{len(figure_chunks)}] Processing {fc.chunk_id} ({fc.paper_id} p.{fc.page} {fc.element_number})...")
        desc = generate_visual_description(fc, client=client, force_refresh=False)
        desc_chunk = build_visual_description_chunk(fc, desc)
        new_desc_chunks.append(desc_chunk)

    # STEP 3: Incremental Indexing
    print(f"\n--- Incremental Indexing of {len(new_desc_chunks)} Visual Description Chunks ---")
    desc_texts = [c.text for c in new_desc_chunks]
    desc_embeddings = embedder.embed_texts(desc_texts)

    vec_index.add_chunks(new_desc_chunks, desc_embeddings)
    bm25_idx.add_chunks(new_desc_chunks)

    # Save updated indices
    vec_index.save(INDEX_DIR)
    bm25_idx.save(INDEX_DIR)

    # Save updated chunks.pkl
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(vec_index.chunks, f)

    after_count = len(vec_index.chunks)
    print(f"Chunks after visual description indexing: {after_count} (added {after_count - before_count})")
    print("Indexing completed successfully!")

if __name__ == "__main__":
    run_visual_pipeline()
