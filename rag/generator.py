# Response generation module
"""
This module builds a prompt from the retrieved passages and the question then have the LLM
answer the question based on the context provided by those passages. 
The prompt is built in a way that encourages the LLM to cite the source of any 
information it uses in its answer.
"""

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
    response = client.message.create(
        model=model,
        max_tokens = 1024,
        system_prompt=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
    )
    return response.content[0].text