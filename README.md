# RAG From Scratch

A retrieval-augmented generation (RAG) chatbot that answers questions using
your own documents instead of the model's memory — with the retrieval core
(chunking, embeddings, vector search) implemented from scratch rather than
wired to an external vector database or embeddings API.

## What this project does

- Splits uploaded documents (`.txt`, `.md`, `.pdf`) into sentence-aware,
  overlapping chunks
- Converts each chunk into a TF-IDF vector using a hand-written vectorizer
  (no pretrained embedding model download required)
- Retrieves the most relevant chunks for a question via cosine-similarity
  search over a NumPy matrix (optionally FAISS for larger corpora)
- Passes the retrieved context — not the raw question alone — into an LLM,
  which reduces hallucinated answers and lets every response be traced back
  to a source chunk
- Ships as a Gradio chat UI, deployable directly to Hugging Face Spaces

## Tech stack

- **Retrieval core (from scratch):** Python, NumPy — chunking, TF-IDF
  embeddings, cosine-similarity vector search
- **Retrieval chain / LLM integration:** LangChain-style prompt
  construction, pluggable OpenAI / Anthropic backends
- **Optional vector backend:** FAISS (`VectorStore(backend="faiss")`)
- **UI / deployment:** Gradio, Hugging Face Spaces

## Project structure

```
rag-from-scratch/
├── app.py                 # Gradio UI
├── rag/
│   ├── chunking.py         # sentence-aware chunker
│   ├── embeddings.py       # from-scratch TF-IDF vectorizer
│   ├── vector_store.py     # cosine-similarity search (+ optional FAISS)
│   ├── llm.py               # pluggable OpenAI / Anthropic generation
│   ├── loaders.py           # .txt / .md / .pdf loaders
│   └── pipeline.py           # ties it all together (RAGPipeline)
├── sample_docs/              # sample files to try the demo with
├── test_pipeline.py            # offline sanity test, no API key needed
├── requirements.txt
└── .env.example
```

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env          # optional: add an OpenAI or Anthropic key
python app.py
```

Without an API key the app still runs end-to-end in **demo mode**, showing
the raw retrieved passages so you can verify retrieval quality on its own.

Verify the retrieval core directly (no UI, no API key needed):

```bash
python test_pipeline.py
```

## Deploying to Hugging Face Spaces

1. Create a new Space → SDK: **Gradio**
2. Push this folder's contents to the Space repo
3. In the Space's **Settings → Repository secrets**, add `OPENAI_API_KEY`
   or `ANTHROPIC_API_KEY`
4. The Space will build automatically from `requirements.txt` and launch
   `app.py`

## Why build the retrieval layer from scratch

Most RAG tutorials call a hosted embeddings API and a managed vector DB,
which hides exactly the part that makes RAG work: turning text into
comparable vectors and searching them efficiently. Implementing TF-IDF and
cosine similarity directly makes the tradeoffs (vocabulary size, why
normalization matters, why chunk overlap matters) visible and swappable —
the same `fit`/`embed` interface can later be pointed at a real
sentence-transformer or embeddings API without touching the rest of the
pipeline.
