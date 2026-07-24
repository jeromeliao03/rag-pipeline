# Response generation module
from .retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite the passages you rely on inline, like [1] or [2]. If the answer is "
    "not contained in the context, say so plainly instead of guessing."
)

def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {chunk.source})\n{chunk.text}")
    context = "\n\n".join(blocks)
    return f"Context passage:\n\n{context}\n\nQuestion: {question}"

def generate_answer(
        question: str,
        chunks: list[RetrievedChunk],
        model:str
) -> str:
    
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
    model=model,
    max_tokens=1024,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": build_user_message(question, chunks)}],
)
    return response.content[0].text