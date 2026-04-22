

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = None
collection = None
client_openai = None


def setup_rag(api_key: str, text: str, persist_directory: str) -> None:
    global chunks, collection, client_openai
    chunks = splitter.split_text(text)

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )

    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="rag_collection",
        embedding_function=embedding_function,
    )

    collection.add(documents=chunks, ids=[f"id_{i}" for i in range(len(chunks))])

    client_openai = OpenAI(api_key=api_key)


def strip_chunk_footer(text: str) -> str:
    """Remove the 'Chunks used' line from an answer before storing it in conversation history."""
    sep = "\n\n---\nChunks used"
    if sep in text:
        return text.split(sep, 1)[0].rstrip()
    return text


def _format_conversation_history(conversation_history: list[dict]) -> str:
    lines = []
    for turn in conversation_history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def rag_query(
    question,
    n_results=5,
    conversation_history: list[dict] | None = None,
    situation: str | None = None,
):
    situation = (situation or "").strip() or None
    # Include situation in the embedding query so retrieval matches the scenario when possible.
    q_for_search = f"Situation:\n{situation}\n\nQuestion:\n{question}" if situation else question
    results = collection.query(
        query_texts=[q_for_search],
        n_results=min(n_results, len(chunks)),
    )
    docs_list = results.get("documents") or []
    ids_list = results.get("ids") or []
    retrieved_docs = docs_list[0] if docs_list else []
    ids_used = ids_list[0] if ids_list else []
    context = "\n\n".join(retrieved_docs)

    if not context:
        return "No relevant documents found."

    hist = conversation_history if conversation_history else []
    hist_block = _format_conversation_history(hist)
    sit_block = (
        f"User-reported conditions (tailor advice only when consistent with the context; do not invent facts):\n{situation}\n\n"
        if situation
        else ""
    )

    if hist_block:
        user_content = (
            f"Context from the knowledge base:\n{context}\n\n"
            f"{sit_block}"
            f"Earlier in this conversation (use only to interpret follow-ups; facts must still match the knowledge base):\n"
            f"{hist_block}\n\n"
            f"Current question: {question}"
        )
    else:
        user_content = f"Context:\n{context}\n\n{sit_block}Question: {question}"

    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer ONLY based on the knowledge base context. Do not invent information. "
                    "You may adapt wording (training, recovery, competition) to the user's conditions when those "
                    "conditions are compatible with the context. "
                    "For follow-up questions, you may use the earlier conversation to understand what the user means, "
                    "but every factual claim must still be supported by the context."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    answer = response.choices[0].message.content
    if ids_used:
        nums = [rid[3:] if isinstance(rid, str) and rid.startswith("id_") else str(rid) for rid in ids_used]
        chunk_note = "\n\n---\nChunks used (index 0, 1, 2, …): " + ", ".join(nums)
        return answer + chunk_note
    return answer
