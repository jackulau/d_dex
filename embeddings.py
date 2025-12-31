"""Embedding generation for RAG system."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False


class EmbeddingGenerator:
    """Generates embeddings for text, code, and images."""

    def __init__(self,
                 text_model: str = 'all-MiniLM-L6-v2',
                 clip_model: str = 'openai/clip-vit-base-patch32',
                 device: Optional[str] = None):
        self.text_model = None
        self.clip_model = None
        self.clip_processor = None

        if device is None:
            self.device = 'cuda' if CLIP_AVAILABLE and torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.text_model = SentenceTransformer(text_model, device=self.device)
            self.text_embedding_dim = self.text_model.get_sentence_embedding_dimension()
        else:
            self.text_embedding_dim = 384

        if CLIP_AVAILABLE and PIL_AVAILABLE:
            self.clip_model = CLIPModel.from_pretrained(clip_model)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
            self.clip_model.to(self.device)
            self.clip_model.eval()
            self.image_embedding_dim = self.clip_model.config.projection_dim
        else:
            self.image_embedding_dim = 512

    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding."""
        if self.text_model is None:
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(self.text_embedding_dim).astype(np.float32)
        return self.text_model.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if self.text_model is None:
            return np.array([self.embed_text(t) for t in texts])
        return self.text_model.encode(
            texts, convert_to_numpy=True, batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        ).astype(np.float32)

    def embed_image(self, img: Union[Image.Image, str, np.ndarray]) -> np.ndarray:
        """Generate image embedding using CLIP."""
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if self.clip_model is None:
            np.random.seed(hash(img.tobytes()) % 2**32)
            return np.random.randn(self.image_embedding_dim).astype(np.float32)

        inputs = self.clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)

        embedding = features / features.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().astype(np.float32)

    def embed_code(self, code: str, language: str = 'unknown') -> np.ndarray:
        """Embed code with language context."""
        return self.embed_text(f"[{language}] {code}")

    def embed_chunk(self, chunk: Dict) -> np.ndarray:
        """Embed chunk based on type."""
        chunk_type = chunk.get('type', 'text')

        if chunk_type == 'text':
            return self.embed_text(chunk['content'])
        elif chunk_type == 'code':
            return self.embed_code(chunk['content'], chunk.get('language', 'unknown'))
        elif chunk_type == 'image':
            filepath = chunk.get('filepath')
            if filepath and Path(filepath).exists():
                return self.embed_image(filepath)
            content = chunk.get('content', '') or chunk.get('description', '')
            return self.embed_text(content)
        else:
            raise ValueError(f"Unknown chunk type: {chunk_type}")

    def embed_chunks(self, chunks: List[Dict], batch_size: int = 32) -> np.ndarray:
        """Embed multiple chunks efficiently."""
        text_chunks = [(i, c) for i, c in enumerate(chunks) if c['type'] == 'text']
        code_chunks = [(i, c) for i, c in enumerate(chunks) if c['type'] == 'code']
        image_chunks = [(i, c) for i, c in enumerate(chunks) if c['type'] == 'image']

        max_dim = max(self.text_embedding_dim, self.image_embedding_dim)
        embeddings = np.zeros((len(chunks), max_dim), dtype=np.float32)

        if text_chunks:
            indices, chunk_list = zip(*text_chunks)
            texts = [c['content'] for c in chunk_list]
            text_embs = self.embed_texts(texts, batch_size)
            for idx, emb in zip(indices, text_embs):
                embeddings[idx, :len(emb)] = emb

        if code_chunks:
            indices, chunk_list = zip(*code_chunks)
            texts = [f"[{c.get('language', 'unknown')}] {c['content']}" for c in chunk_list]
            code_embs = self.embed_texts(texts, batch_size)
            for idx, emb in zip(indices, code_embs):
                embeddings[idx, :len(emb)] = emb

        for idx, chunk in image_chunks:
            emb = self.embed_chunk(chunk)
            embeddings[idx, :len(emb)] = emb

        return embeddings

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def get_embedding_dimension(self, chunk_type: str = 'text') -> int:
        """Get embedding dimension for chunk type."""
        return self.image_embedding_dim if chunk_type == 'image' else self.text_embedding_dim
