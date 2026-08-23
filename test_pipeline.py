"""
Quick sanity check for the RAG pipeline.

Runs fully offline (no API key needed) — proves chunking, embedding, and
retrieval work correctly. The generation step will show demo-mode output
unless OPENAI_API_KEY or ANTHROPIC_API_KEY is set in the environment.
"""

from dotenv import load_dotenv

from rag.pipeline import RAGPipeline
from rag.loaders import load_files

load_dotenv()


def main():
    docs = load_files(
        [
            "sample_docs/company_handbook.txt",
            "sample_docs/product_faq.txt",
        ]
    )

    pipeline = RAGPipeline(chunk_size=500, chunk_overlap=100)
    n_chunks = pipeline.index_documents(docs)
    print(f"Indexed {n_chunks} chunks from {len(docs)} document(s).\n")

    questions = [
        "How many days of paid leave do employees get?",
        "How long does the robot battery last while cleaning?",
        "Can I work fully remote?",
        "Does the robot support Apple HomeKit?",
    ]

    for q in questions:
        print("=" * 70)
        print(f"Q: {q}")
        result = pipeline.ask(q, top_k=3)
        print(f"\nRetrieved {len(result.sources)} chunks:")
        for r in result.sources:
            print(f"  - {r.chunk.source} (chunk {r.chunk.chunk_id}, score={r.score:.3f})")
        print(f"\nA: {result.answer}\n")


if __name__ == "__main__":
    main()
