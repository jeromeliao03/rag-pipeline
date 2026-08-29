"""Answer generation for the RAG pipeline.

This module creates the prompt sent to the LLM, then streams or returns the final
response using the retrieved document chunks as context.
"""

from typing import Iterator

from .retriever import RetrievedChunk

# LLM prompt setup
SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite the passages you rely on inline, like [1] or [2]. If the answer is "
    "not contained in the context, say so plainly instead of guessing. "
    "Write in plain prose — do not use Markdown formatting such as headings (#), "
    "bold (**), bullet lists (-), or horizontal rules (---). "
    "If the answer has distinct parts, separate them into short paragraphs "
    "with a blank line between each. For a section that needs a label, write "
    "it as a short plain line ending in a colon (e.g. 'Detection methods:'), "
    "on its own line, followed by a blank line before the content."
)


# Build prompt from retrieved chunks
def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {chunk.source})\n{chunk.text}")
    context = "\n\n".join(blocks)
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


# Final answer generation
def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
) -> str:
    """Blocking call — returns the complete answer as one string."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
    )
    return response.content[0].text


# Streaming answer generation
def generate_answer_stream(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
) -> Iterator[str]:
    """Streaming call — yields text pieces as the model generates them."""
    import anthropic

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
    ) as stream:
        for text in stream.text_stream:
            yield text