"""
LLM integration layer.

Kept provider-agnostic on purpose: the retrieval half of this project (the
actual "from scratch" work) doesn't care which model generates the final
answer. Drop an API key into .env for either provider and it's used
automatically; with no key at all the app still runs end-to-end in a
"demo mode" that returns the retrieved evidence directly, which is useful
for testing the retrieval quality on its own.
"""

import os

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the answer isn't in the context, say you don't "
    "know instead of guessing. Cite which source each fact comes from."
)


def _build_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


def _call_openai(question: str, context_blocks: list[str], model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(question, context_blocks)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _call_anthropic(question: str, context_blocks: list[str], model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(question, context_blocks)}],
    )
    return response.content[0].text


def _demo_mode_answer(question: str, context_blocks: list[str]) -> str:
    preview = "\n\n".join(f"• {c[:300]}..." for c in context_blocks)
    return (
        "[Demo mode — no OPENAI_API_KEY or ANTHROPIC_API_KEY set, so no LLM "
        "call was made. Showing the raw retrieved context instead. Add a "
        "key to .env to get a generated answer.]\n\n"
        f"Most relevant passages for: \"{question}\"\n\n{preview}"
    )


def generate_answer(
    question: str,
    context_blocks: list[str],
    openai_model: str = "gpt-4o-mini",
    anthropic_model: str = "claude-sonnet-4-5",
) -> str:
    """Routes to whichever provider has a key configured, else demo mode."""
    if os.environ.get("OPENAI_API_KEY"):
        return _call_openai(question, context_blocks, openai_model)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _call_anthropic(question, context_blocks, anthropic_model)
    return _demo_mode_answer(question, context_blocks)
