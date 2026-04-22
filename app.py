

import importlib.util
import os
from pathlib import Path

import gradio as gr

import rag_pipeline as rp
from situation import build_situation

APP_DIR = Path(__file__).resolve().parent
CHROMA_DATA_DIR = APP_DIR / "chroma_ui_data"

# First question + at most this many follow-ups (total turns = 1 + MAX_FOLLOW_UP_QUESTIONS)
MAX_FOLLOW_UP_QUESTIONS = 5


def rag_corpus_txt_path() -> Path:
    return Path(os.environ.get("RAG_CORPUS_PATH", str(APP_DIR / "rag_corpus.txt")))


def on_build(api_key: str):
    txt_file = rag_corpus_txt_path()
    if not txt_file.is_file():
        return f"Missing {txt_file}. Run the notebook cell that writes rag_corpus.txt, then Build index again."
    key = (api_key or "").strip() or os.environ["OPENAI_API_KEY"].strip()
    text = txt_file.read_text(encoding="utf-8", errors="replace")
    rp.setup_rag(key, text, str(CHROMA_DATA_DIR))
    return f"Ready: {len(rp.chunks)} chunks (from {txt_file.name})"


def on_ask(question: str, history: list, city: str, indoor_outdoor: str, surface: str):
    history = list(history or [])
    if not question.strip():
        return "", history
    max_turns = 1 + MAX_FOLLOW_UP_QUESTIONS
    if len(history) >= max_turns:
        return (
            f"Limit reached: {max_turns} questions total (1 initial + {MAX_FOLLOW_UP_QUESTIONS} follow-ups). "
            "Click Clear conversation to start over.",
            history,
        )
    situation = build_situation(city or "", indoor_outdoor or "", surface or "") or None
    raw = rp.rag_query(question.strip(), conversation_history=history, situation=situation)
    answer_for_history = rp.strip_chunk_footer(raw)
    new_history = history + [{"question": question.strip(), "answer": answer_for_history}]
    return raw, new_history


def on_clear():
    return [], ""


def main():
    with gr.Blocks(title="RAG") as d:
        gr.Markdown(
            "# RAG\n"
            "`rag_corpus.txt` next to `app.py`. "
            f"Up to **{MAX_FOLLOW_UP_QUESTIONS}** follow-up questions after the first. "
            "Optional: city (live weather via Open-Meteo), indoor/outdoor, surface — answers stay grounded in your text file.\n\n"
            "*Informational only; not medical advice.*"
        )
        k = gr.Textbox(type="password", label="API key (or OPENAI_API_KEY)")
        st = gr.Textbox(label="Status", interactive=False)
        gr.Button("Build index").click(on_build, [k], st)

        chat_state = gr.State([])

        not_set = "(not set)"
        city = gr.Textbox(label="City / region (optional — for current weather)", lines=1, placeholder="e.g. Boston")
        indoor_outdoor = gr.Dropdown(
            [not_set, "indoor", "outdoor"],
            value=not_set,
            label="Setting",
        )
        surface = gr.Dropdown(
            [not_set, "natural grass", "artificial turf", "hard court", "indoor court", "track", "other"],
            value=not_set,
            label="Surface",
        )
        q = gr.Textbox(label="Question", lines=3)
        a = gr.Textbox(label="Answer", lines=14)
        ask_ins = [q, chat_state, city, indoor_outdoor, surface]
        with gr.Row():
            gr.Button("Ask").click(on_ask, ask_ins, [a, chat_state])
            gr.Button("Clear conversation").click(on_clear, outputs=[chat_state, a])
        q.submit(on_ask, ask_ins, [a, chat_state])

    d.launch(share=importlib.util.find_spec("google.colab") is not None)


if __name__ == "__main__":
    main()
