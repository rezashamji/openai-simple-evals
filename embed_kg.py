"""
Generic embedding-based KG search using KaLM embeddings.

Works with any ARK-compatible knowledge graph (PRIME, MAG, AMAZON, OptmusKG, etc.)

Three independent search modes:
1. Embedding-only: Dense vector similarity (KaLM semantic matching)
2. BM25-only: Keyword matching (Tantivy from ARK)
3. Hybrid: Combines both via Reciprocal Rank Fusion (RRF)

Strategy:
- Part 1: Embed all KG nodes once, save to disk
- Part 2: Embed queries per-request
- Part 3: Search via any of the three modes
"""

from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys
import re


# ============================================================================
# NOTE: clean_query() is copied from ark/src/core/index.py (lines 13-51)
# SOURCE: /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/rshamji/ark/src/core/index.py
#
# This function MUST be kept in sync with ARK's version to ensure embeddings
# use the exact same node name cleaning as the BM25 index. Any changes to
# the original must be reflected here to maintain consistency.
#
# When creating a new code repo, import this from ARK instead of duplicating:
#   from src.core.index import clean_query
# ============================================================================
def clean_query(query: str) -> str:
    """
    Clean query/node names for consistent search across BM25 and embeddings.

    Copied from: ark/src/core/index.py::clean_query()
    """
    # Remove newlines and replace with spaces
    query = query.replace("\n", " ")

    # Remove leading/trailing dashes and spaces
    query = re.sub(r"^[-\s]+", "", query)
    query = re.sub(r"[-\s]+$", "", query)

    # Remove bullet point markers (-, •, *, etc.) at start of segments
    query = re.sub(r"(?:^|\s)[-•*]\s*", " ", query)

    # Remove special characters that can interfere with search
    query = query.replace(":", "")
    query = query.replace('"', "")
    query = query.replace("'", "")
    query = query.replace("(", "")
    query = query.replace(")", "")
    query = query.replace("[", "")
    query = query.replace("]", "")
    query = query.replace("{", "")
    query = query.replace("}", "")
    query = query.replace("<", "")
    query = query.replace(">", "")
    query = query.replace("+", " ")
    query = query.replace("^", "")
    query = query.replace("-", " ")
    query = query.replace("`", " ")

    # Normalize whitespace (multiple spaces -> single space)
    query = re.sub(r"\s+", " ", query)

    # Strip leading/trailing whitespace
    query = query.strip()

    # Escape problematic terms instead of lowercasing
    query = re.sub(r"\b(AND|OR|NOT|DE|IN|THE)\b", r'"\1"', query)

    return query


class KGEmbedder:
    """Generic knowledge graph embedder using KaLM embeddings.

    Works with any KG that has nodes.parquet with columns: [index, name, type, summary]
    Compatible with: PRIME, MAG, AMAZON, OptmusKG, and any other ARK-compatible KG.
    """

    def __init__(self, model_name: str = "tencent/KaLM-Embedding-Gemma3-12B-2511"):
        """Initialize KaLM embedding model."""
        print(f"Loading KaLM model: {model_name}")
        # Use standard attention (not flash_attention_2) to ensure compatibility
        # flash_attention_2 is optional and only provides speed optimization, not accuracy changes
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
            },
        )
        self.model.max_seq_length = 512
        print(f"Model loaded. Max sequence length: {self.model.max_seq_length}")

        self.node_embeddings = None  # (192682, 3840)
        self.nodes_df = None  # DataFrame with [index, name, type, summary]

    def load_nodes(self, nodes_parquet_path: str) -> pd.DataFrame:
        """
        Load KG nodes from parquet file (works with any ARK-compatible KG).

        Expected columns: [index, name, type, summary]
        Examples: PRIME, MAG, AMAZON, OptmusKG all have this format.

        IMPORTANT: Node names are cleaned to match ARK's BM25 index format.
        This ensures consistency between embedding and BM25 search results.
        """
        self.nodes_df = pd.read_parquet(nodes_parquet_path)

        # Clean node names to match ARK's BM25 index format
        # This is critical for consistency in hybrid (RRF) search mode
        original_names = self.nodes_df['name'].copy()
        self.nodes_df['name'] = self.nodes_df['name'].apply(clean_query)

        # Log if any names were changed
        changed = (original_names != self.nodes_df['name']).sum()
        if changed > 0:
            print(f"✓ Cleaned {changed} node names to match BM25 format")
            print(f"  Example cleaning:")
            for i in range(min(5, len(self.nodes_df))):
                if original_names.iloc[i] != self.nodes_df['name'].iloc[i]:
                    print(f"    '{original_names.iloc[i]}' -> '{self.nodes_df['name'].iloc[i]}'")

        print(f"✓ Loaded {len(self.nodes_df)} nodes from {nodes_parquet_path}")
        print(f"  Columns: {list(self.nodes_df.columns)}")
        print(f"  Sample row:")
        print(f"    name: {self.nodes_df.iloc[0]['name']}")
        print(f"    type: {self.nodes_df.iloc[0]['type']}")
        print(f"    summary (first 100 chars): {self.nodes_df.iloc[0]['summary'][:100]}...")
        return self.nodes_df

    def embed_all_nodes(self, batch_size: int = 256) -> np.ndarray:
        """
        Embed all node summaries with KaLM encode_document().

        One-time operation. Save result to disk for reuse across sessions.

        Args:
            batch_size: Batch size for encoding (default 256)

        Returns:
            node_embeddings: (192682, 3840) numpy array
        """
        if self.nodes_df is None:
            raise ValueError("Must call load_nodes() first")

        summaries = self.nodes_df["summary"].tolist()
        print(f"\n✓ Embedding {len(summaries)} node summaries with KaLM...")
        print(f"  Using: model.encode_document()")
        print(f"  Batch size: {batch_size}")

        self.node_embeddings = self.model.encode_document(
            summaries,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print(f"✓ Node embeddings shape: {self.node_embeddings.shape}")
        print(f"  Embedding dimension: {self.node_embeddings.shape[1]}")
        return self.node_embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query with KaLM encode_query().

        Per-query operation, very fast (milliseconds).

        Args:
            query: User question/search query

        Returns:
            query_embedding: (3840,) numpy array
        """
        query_embedding = self.model.encode_query(
            query,
            normalize_embeddings=True,
        )
        return query_embedding

    def search_embedding_only(
        self, query: str, k: int = 10
    ) -> List[Tuple[str, str, str, float]]:
        """
        Dense vector similarity search using embeddings only.

        Good for: Semantic matching, synonyms, conceptual relevance

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            [(node_id, node_type, node_summary, similarity_score), ...]
        """
        if self.node_embeddings is None:
            raise ValueError("Must call embed_all_nodes() first")
        if self.nodes_df is None:
            raise ValueError("Must call load_nodes() first")

        query_embedding = self.embed_query(query)

        # Semantic search: query_embedding vs all node_embeddings
        search_results = util.semantic_search(
            query_embedding, self.node_embeddings, top_k=k
        )

        # Format results
        results = []
        for hit in search_results[0]:
            corpus_id = hit["corpus_id"]
            score = hit["score"]
            node = self.nodes_df.iloc[corpus_id]
            results.append((str(node["name"]), str(node["type"]), str(node["summary"]), score))

        return results

    def reciprocal_rank_fusion(
        self,
        embedding_results: List[Tuple],
        bm25_results: List[Tuple],
        k: int = 10,
        rrf_k: int = 60,
    ) -> List[Tuple[str, str, str, float, Dict]]:
        """
        Combine embedding and BM25 results using Reciprocal Rank Fusion (RRF).

        RRF formula: score = 1/(rrf_k + rank)

        Both rankers contribute equally (no weights needed).

        Args:
            embedding_results: [(node_id, type, summary, embedding_score), ...]
            bm25_results: [(node_id, type, summary, bm25_score), ...]
            k: Number of results to return
            rrf_k: RRF constant (default 60)

        Returns:
            [(node_id, type, summary, fused_score, metadata_dict), ...]
            where metadata contains: embedding_rank, bm25_rank, embedding_score, bm25_score
        """
        print(f"[RRF_DEBUG] Starting RRF fusion with {len(embedding_results)} embedding results, {len(bm25_results)} BM25 results", flush=True, file=sys.stderr)

        # Create lookup: node_id -> (rank, score) for each method
        embedding_lookup = {
            result[0]: (rank + 1, result[3])
            for rank, result in enumerate(embedding_results)
        }
        bm25_lookup = {
            result[0]: (rank + 1, result[3])
            for rank, result in enumerate(bm25_results)
        }

        if embedding_results:
            print(f"[RRF_DEBUG] First embedding result: {embedding_results[0]}", flush=True, file=sys.stderr)
        if bm25_results:
            print(f"[RRF_DEBUG] First BM25 result: {bm25_results[0]}", flush=True, file=sys.stderr)

        # Collect all unique nodes from both searches
        all_nodes = set(embedding_lookup.keys()) | set(bm25_lookup.keys())

        # Apply RRF to each node
        fused_scores = {}
        for node_id in all_nodes:
            rrf_score = 0.0

            # Embedding contribution
            emb_rank, emb_score = embedding_lookup.get(node_id, (rrf_k + 1, 0.0))
            rrf_score += 1.0 / (rrf_k + emb_rank)

            # BM25 contribution
            bm25_rank, bm25_score = bm25_lookup.get(node_id, (rrf_k + 1, 0.0))
            rrf_score += 1.0 / (rrf_k + bm25_rank)

            fused_scores[node_id] = {
                "rrf_score": rrf_score,
                "embedding_rank": emb_rank,
                "bm25_rank": bm25_rank,
                "embedding_score": emb_score,
                "bm25_score": bm25_score,
            }

        # Sort by RRF score and return top-k
        sorted_nodes = sorted(
            fused_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )[:k]

        results = []
        for node_id, metadata in sorted_nodes:
            node_id_str = str(node_id)
            matches = self.nodes_df[self.nodes_df["name"] == node_id_str]
            if len(matches) == 0:
                print(f"[RRF_DEBUG] Node not found in RRF: {node_id_str} (type: {type(node_id)}). Available sample: {list(self.nodes_df['name'].head(3).values)}", flush=True, file=sys.stderr)
                continue
            node = matches.iloc[0]
            results.append(
                (str(node["name"]), str(node["type"]), str(node["summary"]), metadata["rrf_score"], metadata)
            )

        return results

    def save_embeddings(self, output_path: str):
        """
        Save node embeddings to disk for reuse.

        Args:
            output_path: Path to save .npy file
        """
        if self.node_embeddings is None:
            raise ValueError("Must call embed_all_nodes() first")
        np.save(output_path, self.node_embeddings)
        size_mb = self.node_embeddings.nbytes / (1024 * 1024)
        print(f"✓ Saved embeddings to {output_path} ({size_mb:.1f} MB)")

    def load_embeddings(self, embeddings_path: str):
        """
        Load pre-computed embeddings from disk.

        Skips the expensive embedding step if already computed.

        Args:
            embeddings_path: Path to .npy file
        """
        self.node_embeddings = np.load(embeddings_path)
        print(f"✓ Loaded embeddings from {embeddings_path}")
        print(f"  Shape: {self.node_embeddings.shape}")


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Embed all nodes in a knowledge graph using KaLM"
    )
    parser.add_argument(
        "--graph-path",
        required=True,
        type=str,
        help="Path to graph directory (must contain nodes.parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for embeddings (default: {graph-path}/embeddings_kalm.npy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding (default: 256)",
    )

    args = parser.parse_args()

    from pathlib import Path

    graph_path = Path(args.graph_path)
    nodes_parquet = graph_path / "nodes.parquet"

    # Set output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = graph_path / "embeddings_kalm.npy"

    # Validate inputs
    if not graph_path.exists():
        print(f"ERROR: Graph path does not exist: {graph_path}")
        exit(1)

    if not nodes_parquet.exists():
        print(f"ERROR: nodes.parquet not found at: {nodes_parquet}")
        exit(1)

    # Create embedder and embed
    print(f"\n{'='*70}")
    print(f"EMBEDDING KNOWLEDGE GRAPH NODES")
    print(f"{'='*70}")
    print(f"Graph: {graph_path}")
    print(f"Output: {output_path}")

    embedder = KGEmbedder(model_name="tencent/KaLM-Embedding-Gemma3-12B-2511")

    print(f"\n[1/3] Loading nodes from parquet...")
    nodes_df = embedder.load_nodes(str(nodes_parquet))

    print(f"\n[2/3] Embedding {len(nodes_df)} nodes with KaLM (batch_size={args.batch_size})...")
    embeddings = embedder.embed_all_nodes(batch_size=args.batch_size)

    print(f"\n[3/3] Saving embeddings to disk...")
    embedder.save_embeddings(str(output_path))

    print(f"\n{'='*70}")
    print(f"✓ EMBEDDING COMPLETE")
    print(f"{'='*70}")
    print(f"Embeddings saved to: {output_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Size: {embeddings.nbytes / 1e9:.2f} GB")
    print("  from ark.src.core.index import GraphIndex")
    print("  graph_index = GraphIndex(path='...')")
    print("  bm25_nodes, bm25_scores = graph_index.search(query, k=10)")
    print()

    print("Hybrid fusion would combine both rankings via RRF:")
    print("  fused = embedder.reciprocal_rank_fusion(")
    print("      embedding_results=emb_results,")
    print("      bm25_results=bm25_results,")
    print("      k=10")
    print("  )")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✓ KGEmbedder class provides three search modes:")
    print("  1. search_embedding_only() - Dense vector similarity")
    print("  2. search_bm25() - (integrate with ARK's GraphIndex)")
    print("  3. search_hybrid() - RRF combination of both")
    print()
    print("✓ Embeddings saved to:", output_path)
    print("✓ Ready to integrate with ARK's BM25 retrieval pipeline")
    print()
