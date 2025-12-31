"""Document chunking for RAG system."""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class DocumentChunker:
    """Processes documents into chunks for embedding and retrieval."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_markdown(self, filepath: str) -> str:
        """Load markdown file content."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_text(self, filepath: str) -> str:
        """Load plain text file content."""
        return self.load_markdown(filepath)

    def extract_code_blocks(self, text: str) -> Tuple[List[Dict], str]:
        """Extract fenced code blocks from markdown."""
        pattern = r'```(\w+)?\n(.*?)```'
        code_blocks = []

        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or 'unknown'
            code = match.group(2).strip()
            if not code:
                continue
            code_blocks.append({
                'type': 'code',
                'language': language,
                'content': code,
                'start_pos': match.start(),
                'end_pos': match.end()
            })

        text_only = re.sub(pattern, '[CODE_BLOCK]', text, flags=re.DOTALL)
        return code_blocks, text_only

    def extract_headers(self, text: str) -> List[Dict]:
        """Extract markdown headers."""
        headers = []
        pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(pattern, text, re.MULTILINE):
            headers.append({
                'level': len(match.group(1)),
                'content': match.group(2).strip(),
                'position': match.start()
            })
        return headers

    def chunk_text(self, text: str) -> List[Dict]:
        """Chunk text with overlap using word boundaries."""
        text = ' '.join(text.split())
        words = text.split()
        if not words:
            return []

        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < self.overlap and i > 0:
                continue
            chunks.append({
                'type': 'text',
                'content': ' '.join(chunk_words),
                'chunk_id': len(chunks),
                'word_start': i,
                'word_end': i + len(chunk_words)
            })
        return chunks

    def chunk_by_sentences(self, text: str, max_chunk_size: int = 1000) -> List[Dict]:
        """Chunk text using sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_length + len(sentence) > max_chunk_size and current_chunk:
                chunks.append({
                    'type': 'text',
                    'content': ' '.join(current_chunk),
                    'chunk_id': len(chunks)
                })
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence) + 1

        if current_chunk:
            chunks.append({
                'type': 'text',
                'content': ' '.join(current_chunk),
                'chunk_id': len(chunks)
            })
        return chunks

    def process_document(self, filepath: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Process document into chunks with metadata."""
        content = self.load_markdown(filepath)
        headers = self.extract_headers(content)
        code_blocks, text_only = self.extract_code_blocks(content)
        text_chunks = self.chunk_text(text_only)
        all_chunks = text_chunks + code_blocks

        for chunk in all_chunks:
            chunk['source'] = str(filepath)
            chunk['metadata'] = metadata or {}

            pos = chunk.get('start_pos', chunk.get('word_start', 0) * 6)
            relevant_header = None
            for header in headers:
                if header['position'] <= pos:
                    relevant_header = header
                else:
                    break
            if relevant_header:
                chunk['section'] = relevant_header['content']

        return all_chunks

    def process_directory(self, dirpath: str, pattern: str = "*.md",
                         metadata: Optional[Dict] = None) -> List[Dict]:
        """Process all matching documents in a directory."""
        path = Path(dirpath)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")

        all_chunks = []
        for filepath in path.glob(pattern):
            doc_metadata = {**(metadata or {}), 'filename': filepath.name}
            chunks = self.process_document(str(filepath), doc_metadata)
            all_chunks.extend(chunks)
        return all_chunks
