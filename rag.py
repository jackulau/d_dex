"""RAG system with OpenRouter integration."""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.parse import urljoin, urlparse

from chunker import DocumentChunker
from image_processor import ImageProcessor
from embeddings import EmbeddingGenerator
from vector_store import VectorStore, SearchResult

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_MODELS = {
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-sonnet": "anthropic/claude-3-sonnet",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gpt-4": "openai/gpt-4",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "gemini-pro": "google/gemini-pro",
    "gemini-pro-1.5": "google/gemini-pro-1.5",
    "mistral-large": "mistralai/mistral-large",
    "mistral-medium": "mistralai/mistral-medium",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
    "deepseek-coder": "deepseek/deepseek-coder",
    "qwen-72b": "qwen/qwen-72b-chat",
}


class RAGSystem:
    """Complete RAG system with OpenRouter LLM integration."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "anthropic/claude-3.5-sonnet",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 embedding_dimension: int = 384,
                 site_url: Optional[str] = None,
                 site_name: Optional[str] = None):
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.image_processor = ImageProcessor()
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore(dimension=embedding_dimension)

        self.model = self._resolve_model_name(model)
        self.site_url = site_url
        self.site_name = site_name
        self._setup_openrouter(api_key)

        self.stats = {
            'documents_ingested': 0,
            'images_ingested': 0,
            'total_chunks': 0,
            'queries_processed': 0
        }

    def _resolve_model_name(self, model: str) -> str:
        """Resolve short model names to full paths."""
        return OPENROUTER_MODELS.get(model, model)

    def _setup_openrouter(self, api_key: Optional[str]):
        """Set up OpenRouter client."""
        if not OPENAI_AVAILABLE:
            self.client = None
            return

        api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            self.client = None
            return

        self.client = openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def set_model(self, model: str):
        """Change the LLM model."""
        self.model = self._resolve_model_name(model)

    def list_models(self) -> Dict[str, str]:
        """List available model shortcuts."""
        return OPENROUTER_MODELS.copy()

    def ingest_document(self, filepath: str, metadata: Optional[Dict] = None) -> int:
        """Ingest a document."""
        chunks = self.chunker.process_document(filepath, metadata)
        for chunk in chunks:
            embedding = self.embedder.embed_chunk(chunk)
            self.vector_store.add(embedding, chunk)
        self.stats['documents_ingested'] += 1
        self.stats['total_chunks'] += len(chunks)
        return len(chunks)

    def ingest_directory(self, dirpath: str, pattern: str = "*.md",
                        metadata: Optional[Dict] = None) -> int:
        """Ingest all documents in a directory."""
        chunks = self.chunker.process_directory(dirpath, pattern, metadata)
        for chunk in chunks:
            embedding = self.embedder.embed_chunk(chunk)
            self.vector_store.add(embedding, chunk)
        self.stats['documents_ingested'] += len(set(c['source'] for c in chunks))
        self.stats['total_chunks'] += len(chunks)
        return len(chunks)

    def ingest_image(self, filepath: str, metadata: Optional[Dict] = None) -> int:
        """Ingest an image."""
        try:
            chunk = self.image_processor.process_image(filepath, metadata)
            embedding = self.embedder.embed_chunk(chunk)
            self.vector_store.add(embedding, chunk)
            self.stats['images_ingested'] += 1
            self.stats['total_chunks'] += 1
            return 1
        except Exception:
            return 0

    def ingest_text(self, text: str, source: str = "direct_input",
                   metadata: Optional[Dict] = None) -> int:
        """Ingest raw text."""
        chunks = self.chunker.chunk_text(text)
        for chunk in chunks:
            chunk['source'] = source
            chunk['metadata'] = metadata or {}
            embedding = self.embedder.embed_chunk(chunk)
            self.vector_store.add(embedding, chunk)
        self.stats['total_chunks'] += len(chunks)
        return len(chunks)

    def ingest_url(self, url: str, metadata: Optional[Dict] = None) -> int:
        """Ingest content from a URL."""
        if not WEB_AVAILABLE:
            raise ImportError("Install requests and beautifulsoup4: pip install requests beautifulsoup4")

        response = requests.get(url, headers={'User-Agent': 'RAG-Bot/1.0'}, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, nav, footer elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        # Extract text from main content areas
        main = soup.find('main') or soup.find('article') or soup.find('body')
        text = main.get_text(separator='\n', strip=True) if main else soup.get_text()

        # Extract code blocks
        code_blocks = []
        for pre in soup.find_all('pre'):
            code = pre.get_text(strip=True)
            lang = 'unknown'
            if pre.find('code'):
                classes = pre.find('code').get('class', [])
                for cls in classes:
                    if cls.startswith('language-'):
                        lang = cls.replace('language-', '')
                        break
            code_blocks.append({'type': 'code', 'language': lang, 'content': code})

        # Chunk the text
        chunks = self.chunker.chunk_text(text)
        all_chunks = chunks + code_blocks

        for chunk in all_chunks:
            chunk['source'] = url
            chunk['metadata'] = {**(metadata or {}), 'url': url}
            embedding = self.embedder.embed_chunk(chunk)
            self.vector_store.add(embedding, chunk)

        self.stats['documents_ingested'] += 1
        self.stats['total_chunks'] += len(all_chunks)
        return len(all_chunks)

    def ingest_website(self, base_url: str, max_pages: int = 20,
                      metadata: Optional[Dict] = None) -> int:
        """Crawl and ingest a website (follows internal links)."""
        if not WEB_AVAILABLE:
            raise ImportError("Install requests and beautifulsoup4: pip install requests beautifulsoup4")

        visited = set()
        to_visit = [base_url]
        total_chunks = 0
        base_domain = urlparse(base_url).netloc

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue

            try:
                print(f"  Fetching: {url}")
                chunks = self.ingest_url(url, metadata)
                total_chunks += chunks
                visited.add(url)

                # Find more links
                response = requests.get(url, headers={'User-Agent': 'RAG-Bot/1.0'}, timeout=30)
                soup = BeautifulSoup(response.text, 'html.parser')

                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)

                    # Only follow internal links
                    if parsed.netloc == base_domain and full_url not in visited:
                        # Skip anchors, files, etc
                        if not any(full_url.endswith(ext) for ext in ['.pdf', '.png', '.jpg', '.zip']):
                            if '#' not in full_url:
                                to_visit.append(full_url.split('?')[0])

            except Exception as e:
                print(f"  Error on {url}: {e}")

        return total_chunks

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict] = None,
                use_hybrid: bool = False) -> List[SearchResult]:
        """Retrieve relevant chunks."""
        query_embedding = self.embedder.embed_text(query)
        if use_hybrid:
            return self.vector_store.search_hybrid(query_embedding, query, top_k)
        return self.vector_store.search(query_embedding, top_k, filters)

    def format_context(self, results: List[SearchResult]) -> str:
        """Format chunks into context string."""
        parts = []
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            score = result.score
            chunk_type = chunk.get('type', 'text')

            if chunk_type == 'text':
                parts.append(f"[Source {i} - Text - Score: {score:.3f}]\n{chunk['content']}\n")
            elif chunk_type == 'code':
                lang = chunk.get('language', 'unknown')
                parts.append(f"[Source {i} - Code ({lang}) - Score: {score:.3f}]\n```{lang}\n{chunk['content']}\n```\n")
            elif chunk_type == 'image':
                parts.append(
                    f"[Source {i} - Image - Score: {score:.3f}]\n"
                    f"File: {chunk.get('filepath', 'unknown')}\n"
                    f"Description: {chunk.get('description', 'N/A')}\n"
                    f"Text: {chunk.get('content', 'None')[:500]}\n"
                )
        return "\n---\n".join(parts)

    def generate_response(self, query: str, context: str, model: Optional[str] = None,
                         system_prompt: Optional[str] = None, temperature: float = 0.7,
                         max_tokens: int = 1000) -> str:
        """Generate LLM response."""
        if self.client is None:
            return f"[LLM not configured. Set OPENROUTER_API_KEY.]\n\nContext:\n{context}"

        if system_prompt is None:
            system_prompt = (
                "You are a helpful technical documentation assistant. "
                "Use the provided context to answer accurately. "
                "Include code examples when relevant. "
                "Say if you cannot find the answer in the context."
            )

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        use_model = self._resolve_model_name(model) if model else self.model

        try:
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                extra_headers["X-Title"] = self.site_name

            response = self.client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers if extra_headers else None
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenRouter error: {e}]"

    def query(self, question: str, top_k: int = 5, filters: Optional[Dict] = None,
             use_hybrid: bool = False, model: Optional[str] = None) -> Dict[str, Any]:
        """Full RAG pipeline: retrieve and generate."""
        self.stats['queries_processed'] += 1
        results = self.retrieve(question, top_k, filters, use_hybrid)

        if not results:
            return {
                'question': question,
                'answer': "No relevant information found.",
                'sources': [],
                'scores': [],
                'model': model or self.model
            }

        context = self.format_context(results)
        answer = self.generate_response(question, context, model)

        return {
            'question': question,
            'answer': answer,
            'sources': [r.chunk for r in results],
            'scores': [r.score for r in results],
            'context': context,
            'model': self._resolve_model_name(model) if model else self.model
        }

    def save(self, filepath: str):
        """Save vector store."""
        self.vector_store.save(filepath)

    def load(self, filepath: str):
        """Load vector store."""
        self.vector_store.load(filepath)

    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            **self.stats,
            'current_model': self.model,
            'vector_store': self.vector_store.get_stats()
        }

    def clear(self):
        """Clear all data."""
        self.vector_store.clear()
        self.stats = {
            'documents_ingested': 0,
            'images_ingested': 0,
            'total_chunks': 0,
            'queries_processed': 0
        }
