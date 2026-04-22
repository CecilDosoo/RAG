!pip install chromadb openai langchain-text-splitters tiktoken


#Cell 1
from google.colab import drive
drive.mount('/content/drive')


#Cell 2
import os
os.environ["OPENAI_API_KEY"] = "your-key"
api_key = os.environ["OPENAI_API_KEY"]


#Cell 3
from google.colab import files
up_files = files.upload()

text = ""
for filename, file_content in up_files.items():
    if filename.endswith('.txt'):
        text += file_content.decode('utf-8') + "\n"


#Cell 4 — save corpus for Gradio app.py (same folder as rag_pipeline.py / app.py on Drive)
PROJECT = "/content/drive/MyDrive/rag_chroma_db"

from pathlib import Path
Path(PROJECT).mkdir(parents=True, exist_ok=True)
(Path(PROJECT) / "rag_corpus.txt").write_text(text, encoding="utf-8")


#Cell 5
import sys
sys.path.insert(0, PROJECT)


#Cell 6
from rag_pipeline import setup_rag, rag_query

persist_directory = "/content/drive/MyDrive/chroma_db"

setup_rag(api_key, text, persist_directory)

question = "Your question here"
answer = rag_query(question)
print(answer)
