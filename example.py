#!/usr/bin/env python3
"""RAG system interactive example."""

from pathlib import Path
from rag import RAGSystem, OPENROUTER_MODELS


def create_sample_docs():
    """Create sample documentation for testing."""
    docs_dir = Path("/tmp/rag_docs")
    docs_dir.mkdir(exist_ok=True)

    (docs_dir / "react.md").write_text("""
# React Hooks

## useState

```jsx
const [count, setCount] = useState(0);
```

useState returns a stateful value and a function to update it.

## useEffect

```jsx
useEffect(() => {
    document.title = `Count: ${count}`;
}, [count]);
```

useEffect runs side effects after render.
""")

    (docs_dir / "auth.md").write_text("""
# Authentication

## OAuth Flow

```javascript
async function login() {
    const response = await fetch('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username, password })
    });
    return response.json();
}
```

## Token Refresh

```javascript
async function refresh(token) {
    const response = await fetch('/api/auth/refresh', {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    return response.json();
}
```
""")

    return docs_dir


def main():
    print("=" * 50)
    print("RAG System")
    print("=" * 50)

    # Create sample docs
    docs_dir = create_sample_docs()

    # Initialize
    print("\nInitializing...")
    rag = RAGSystem(model="gpt-4o-mini", chunk_size=200)

    # Ingest
    print("Ingesting documents...")
    rag.ingest_directory(str(docs_dir), pattern="*.md")
    print(f"Indexed {rag.stats['total_chunks']} chunks")

    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("Commands: /model <name>, /models, /quit")
    print("=" * 50)

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue

            if query == "/quit":
                break

            if query == "/models":
                print("\nAvailable models:")
                for short, full in OPENROUTER_MODELS.items():
                    marker = " *" if full == rag.model else ""
                    print(f"  {short:20} {full}{marker}")
                continue

            if query.startswith("/model "):
                rag.set_model(query[7:].strip())
                print(f"Model: {rag.model}")
                continue

            # Query
            result = rag.query(query, top_k=3, use_hybrid=True)
            print(f"\n[{result['model']}]")
            print(result['answer'])

            if result['sources']:
                print("\nSources:")
                for src, score in zip(result['sources'][:3], result['scores'][:3]):
                    name = Path(src.get('source', '')).name
                    print(f"  - {name} ({score:.2f})")

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
