"""
RAG From Scratch — Gradio app.

Upload documents, then ask questions about them. The app retrieves the most
relevant passages using a from-scratch TF-IDF + cosine-similarity search
before generating an answer, so responses stay grounded in your documents
instead of the model's own memory.
"""

import os

import gradio as gr
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline
from rag.loaders import load_files

# Hugging Face ZeroGPU Spaces require at least one function decorated with
# @spaces.GPU to be detected at startup. This app is pure CPU/Python and
# never needs a GPU, so we fall back to a harmless no-op decorator when the
# `spaces` package isn't available (e.g. running locally).
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(func):
        return func

load_dotenv()

pipeline = RAGPipeline(chunk_size=800, chunk_overlap=150)
state = {"indexed": False, "n_chunks": 0, "n_docs": 0}


def index_documents(files):
    if not files:
        return "⚠️ Please upload at least one .txt, .md, or .pdf file first."

    try:
        paths = [f.name if hasattr(f, "name") else f for f in files]
        docs = load_files(paths)
        n_chunks = pipeline.index_documents(docs)
        state["indexed"] = True
        state["n_chunks"] = n_chunks
        state["n_docs"] = len(docs)
        return (
            f"✅ Indexed {len(docs)} document(s) into {n_chunks} chunks. "
            "You can start asking questions below."
        )
    except Exception as e:
        return f"❌ Failed to index documents: {e}"


@gpu_decorator
def answer_question(question, top_k, history):
    if not question or not question.strip():
        return history, ""
    if not state["indexed"]:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "⚠️ Please upload and index documents first (above)."},
        ]
        return history, ""

    result = pipeline.ask(question, top_k=int(top_k))
    sources_line = ", ".join(
        f"{r.chunk.source}#{r.chunk.chunk_id} ({r.score:.2f})" for r in result.sources
    )
    reply = f"{result.answer}\n\n---\n**Retrieved from:** {sources_line}"
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


def load_sample_docs():
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    paths = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir)]
    return index_documents(paths)


with gr.Blocks(title="RAG From Scratch") as demo:
    gr.Markdown(
        """
        # 🔍 RAG From Scratch
        A retrieval-augmented chatbot built without any external vector-DB
        or embedding API — chunking, TF-IDF embeddings, and cosine-similarity
        search are all implemented from scratch. Upload documents, then ask
        questions grounded in their content.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                file_count="multiple",
                file_types=[".txt", ".md", ".pdf"],
                label="Upload documents",
            )
            index_btn = gr.Button("📚 Index documents", variant="primary")
            sample_btn = gr.Button("Or try the sample docs")
            status = gr.Textbox(label="Status", interactive=False)
            top_k = gr.Slider(1, 8, value=4, step=1, label="Chunks to retrieve (top-k)")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Ask about your documents", height=430)
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. How many days of paid leave do employees get?",
            )
            ask_btn = gr.Button("Ask", variant="primary")

    index_btn.click(index_documents, inputs=[file_input], outputs=[status])
    sample_btn.click(load_sample_docs, outputs=[status])
    ask_btn.click(answer_question, inputs=[question, top_k, chatbot], outputs=[chatbot, question])
    question.submit(answer_question, inputs=[question, top_k, chatbot], outputs=[chatbot, question])


if __name__ == "__main__":
    demo.launch()
