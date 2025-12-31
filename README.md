# RAG System

A complete Retrieval-Augmented Generation system built from scratch in Python.

## Architecture

```
Document → Chunking → Embedding → Vector Store → Retrieval → LLM Generation
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from rag import RAGSystem

# Initialize (set OPENROUTER_API_KEY env var for LLM generation)
rag = RAGSystem(model="gpt-4o-mini")

# Ingest from a docs website
rag.ingest_url("https://react.dev/learn")

# Or crawl multiple pages
rag.ingest_website("https://docs.example.com", max_pages=20)

# Or use local files
rag.ingest_directory("docs/", pattern="*.md")

# Query
result = rag.query("How do I use hooks?")
print(result['answer'])
```

## Components

| File | Description |
|------|-------------|
| `chunker.py` | Splits documents into chunks, extracts code blocks |
| `embeddings.py` | Generates vector embeddings (sentence-transformers) |
| `vector_store.py` | Stores vectors, performs similarity search |
| `image_processor.py` | Processes images for CLIP embeddings |
| `rag.py` | Main system with OpenRouter LLM integration |

## Configuration

### API Key

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

Or pass directly:

```python
rag = RAGSystem(api_key="sk-or-v1-...")
```

### Models

Use any OpenRouter model with short names or full paths:

```python
rag = RAGSystem(model="gpt-4o")           # Short name
rag = RAGSystem(model="openai/gpt-4o")    # Full path

# Change model
rag.set_model("claude-3.5-sonnet")

# Per-query override
result = rag.query("question", model="gpt-4o-mini")
```

**Available shortcuts:**

| Short | Full Path |
|-------|-----------|
| `gpt-4o` | `openai/gpt-4o` |
| `gpt-4o-mini` | `openai/gpt-4o-mini` |
| `claude-3.5-sonnet` | `anthropic/claude-3.5-sonnet` |
| `claude-3-haiku` | `anthropic/claude-3-haiku` |
| `llama-3.1-70b` | `meta-llama/llama-3.1-70b-instruct` |
| `gemini-pro-1.5` | `google/gemini-pro-1.5` |
| `mixtral-8x7b` | `mistralai/mixtral-8x7b-instruct` |

## Usage

### Ingest Documents

```python
# Single file
rag.ingest_document("docs/api.md", metadata={"topic": "API"})

# Directory
rag.ingest_directory("docs/", pattern="*.md")

# Raw text
rag.ingest_text("Some documentation text...", source="manual")

# Single URL
rag.ingest_url("https://docs.example.com/getting-started")

# Crawl entire website (follows links)
rag.ingest_website("https://docs.example.com", max_pages=50)
```

### Query

```python
# Basic query
result = rag.query("How do I deploy?")

# With options
result = rag.query(
    "How do I deploy?",
    top_k=5,              # Number of chunks to retrieve
    use_hybrid=True,      # Combine vector + keyword search
    model="gpt-4o-mini"   # Override model
)

print(result['answer'])
print(result['sources'])  # Retrieved chunks
print(result['scores'])   # Similarity scores
```

### Filtered Search

```python
# Only code blocks
results = rag.retrieve("auth example", filters={'type': 'code'})

# By metadata
results = rag.retrieve("login", filters={'metadata': {'topic': 'Auth'}})
```

### Save / Load

```python
rag.save("vectors.pkl")

# Later...
rag = RAGSystem()
rag.load("vectors.pkl")
```

## How It Works

1. **Chunking**: Documents split into overlapping text chunks + extracted code blocks
2. **Embedding**: Each chunk converted to 384-dim vector via sentence-transformers
3. **Storage**: Vectors stored in memory with metadata
4. **Retrieval**: Query embedded, top-k similar chunks found via cosine similarity
5. **Generation**: Retrieved context + question sent to LLM via OpenRouter

## Dependencies

- `numpy` - Vector operations
- `Pillow` - Image processing
- `sentence-transformers` - Text embeddings
- `transformers` - Model loading
- `torch` - Neural network backend
- `openai` - OpenRouter API client
- `requests` - Web fetching
- `beautifulsoup4` - HTML parsing

## Example

```bash
# Set API key
export OPENROUTER_API_KEY=sk-or-v1-...

# Run example
python example.py
```

Interactive mode:
```
Commands: /model <name>, /models, /quit
You: How do I use useState?
[openai/gpt-4o-mini]
useState is a React hook that...
```
