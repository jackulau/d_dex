"""Vector store for RAG system."""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import heapq


@dataclass
class SearchResult:
    """Search result container."""
    chunk: Dict
    score: float
    index: int


class VectorStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.chunks: List[Dict] = []
        self.index = 0

    def add(self, embedding: np.ndarray, chunk: Dict) -> int:
        """Add vector and chunk to store."""
        if len(embedding) != self.dimension:
            if len(embedding) < self.dimension:
                embedding = np.pad(embedding, (0, self.dimension - len(embedding)))
            else:
                embedding = embedding[:self.dimension]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self.vectors.append(embedding.astype(np.float32))
        chunk['id'] = self.index
        self.chunks.append(chunk)
        idx = self.index
        self.index += 1
        return idx

    def add_batch(self, embeddings: np.ndarray, chunks: List[Dict]) -> List[int]:
        """Add multiple vectors and chunks."""
        return [self.add(emb, chunk) for emb, chunk in zip(embeddings, chunks)]

    def _matches_filters(self, chunk: Dict, filters: Dict) -> bool:
        """Check if chunk matches filters."""
        for key, value in filters.items():
            if key == 'type' and chunk.get('type') != value:
                return False
            elif key == 'language' and chunk.get('language') != value:
                return False
            elif key == 'source':
                source = chunk.get('source', '')
                if not source.endswith(value) and value not in source:
                    return False
            elif key == 'metadata':
                chunk_meta = chunk.get('metadata', {})
                for mk, mv in value.items():
                    if chunk_meta.get(mk) != mv:
                        return False
            elif chunk.get(key) != value:
                return False
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[Dict] = None, threshold: float = 0.0) -> List[SearchResult]:
        """Search for similar vectors."""
        if not self.vectors:
            return []

        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        vectors_array = np.array(self.vectors)
        similarities = np.dot(vectors_array, query_embedding)

        results = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                continue
            if filters and not self._matches_filters(self.chunks[i], filters):
                continue
            if len(results) < top_k:
                heapq.heappush(results, (sim, i))
            elif sim > results[0][0]:
                heapq.heapreplace(results, (sim, i))

        return [
            SearchResult(chunk=self.chunks[idx], score=float(sim), index=idx)
            for sim, idx in sorted(results, reverse=True)
        ]

    def search_hybrid(self, query_embedding: np.ndarray, keyword_query: str,
                     top_k: int = 5, alpha: float = 0.7) -> List[SearchResult]:
        """Hybrid search with vector + keyword matching."""
        vector_results = self.search(query_embedding, top_k * 2)
        keywords = keyword_query.lower().split()

        hybrid_scores = []
        for result in vector_results:
            content = result.chunk.get('content', '').lower()
            keyword_matches = sum(1 for kw in keywords if kw in content)
            keyword_score = keyword_matches / len(keywords) if keywords else 0
            hybrid_score = alpha * result.score + (1 - alpha) * keyword_score
            hybrid_scores.append((hybrid_score, result))

        hybrid_scores.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(chunk=r.chunk, score=score, index=r.index)
            for score, r in hybrid_scores[:top_k]
        ]

    def get(self, index: int) -> Optional[Tuple[np.ndarray, Dict]]:
        """Get vector and chunk by index."""
        if 0 <= index < len(self.vectors):
            return self.vectors[index], self.chunks[index]
        return None

    def delete(self, index: int) -> bool:
        """Soft delete a vector."""
        if 0 <= index < len(self.chunks):
            self.chunks[index]['_deleted'] = True
            return True
        return False

    def update(self, index: int, embedding: Optional[np.ndarray] = None,
               chunk: Optional[Dict] = None) -> bool:
        """Update vector and/or chunk."""
        if not (0 <= index < len(self.vectors)):
            return False
        if embedding is not None:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            self.vectors[index] = embedding.astype(np.float32)
        if chunk is not None:
            chunk['id'] = index
            self.chunks[index] = chunk
        return True

    def __len__(self) -> int:
        return len(self.vectors)

    def save(self, filepath: str):
        """Save store to disk."""
        data = {
            'dimension': self.dimension,
            'vectors': [v.tolist() for v in self.vectors],
            'chunks': self.chunks,
            'index': self.index
        }
        path = Path(filepath)
        if path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    def load(self, filepath: str):
        """Load store from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)

        self.dimension = data['dimension']
        self.vectors = [np.array(v, dtype=np.float32) for v in data['vectors']]
        self.chunks = data['chunks']
        self.index = data['index']

    def get_stats(self) -> Dict:
        """Get store statistics."""
        if not self.vectors:
            return {'total_vectors': 0, 'dimension': self.dimension}

        type_counts = {}
        for chunk in self.chunks:
            t = chunk.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1

        deleted = sum(1 for c in self.chunks if c.get('_deleted', False))

        return {
            'total_vectors': len(self.vectors),
            'active_vectors': len(self.vectors) - deleted,
            'deleted_vectors': deleted,
            'dimension': self.dimension,
            'type_counts': type_counts,
            'memory_bytes': sum(v.nbytes for v in self.vectors)
        }

    def clear(self):
        """Clear all data."""
        self.vectors = []
        self.chunks = []
        self.index = 0
