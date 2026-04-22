# RAG (Implemented through Colab)
A retrieval-augmented generation (RAG) app for question answering over a custom text corpus, using ChromaDB (persistent vector store), OpenAI embeddings (text-embedding-3-small) and GPT-4o-mini, with an optional “situation” layer (live weather via Open-Meteo, plus indoor/outdoor and surface) to steer retrieval and answers.

##Rag_pipeline.py

Splits text, embeds into Chroma, runs similarity search, calls the chat model with strict “only from context” system instructions, supports conversation history and situation text, and appends chunk id footers to answers.

##app.py

Gradio app: load rag_corpus.txt, build index with API key, chat with a cap on follow-ups, optional city / setting / surface.

##Situatio.py

Builds the situation string (weather + setting + surface).

##main.py

Colab-oriented flow: mount Drive, upload txts, write rag_corpus.txt, call setup_rag / rag_query.
