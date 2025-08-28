# 📘 PDF Question‑Answering Bot (RAG with LangChain + Gemini)

A simple **Retrieval‑Augmented Generation (RAG)** app that answers questions about an uploaded PDF using:
- **LangChain** (parsing, splitting, retrieval)
- **Google Gemini 1.5 Flash** (LLM via `langchain-google-genai`)
- **HuggingFace MiniLM** embeddings
- **Chroma** vector store
- **Gradio** web UI with a custom blue theme

---

## ✨ ScreenShot
<img width="1284" height="657" alt="Screenshot 2025-08-28 at 9 12 17 AM" src="https://github.com/user-attachments/assets/a61f4d80-f58e-40ce-ba87-25c17fe8055f" />



---

## ✨ Features
- Upload a PDF and ask natural‑language questions about its contents.
- Contextual compression retriever to reduce irrelevant text before the LLM.
- Grounded answers that say *“I don't know based on the document.”* when info is missing.
- Clean, responsive Gradio UI with a page footer: **©2025 Ali Maqsood — All Rights Reserved**.

---

## 📁 Project Structure
```
MY QA BOT/
├── my_env/                 # local virtualenv (ignored)
├── .env                    # holds GEMINI_KEY (ignored)
├── qabot.py                # main app
├── sample_text.pdf         # example doc
└── requirements.txt        # dependencies
```

> Add a `.gitignore` (recommended):
```
my_env/
.env
__pycache__/
*.pyc
```

---

## 🔧 Setup

### 1) Create & activate a virtual environment
```bash
python3 -m venv my_env
source my_env/bin/activate      # macOS/Linux
# my_env\Scripts\activate     # Windows (PowerShell)
```

### 2) Install deps
```bash
pip install -r requirements.txt
```

### 3) Configure environment
Create a file named **.env** in the project root:
```
GEMINI_KEY=your_google_gemini_api_key_here
```

---

## ▶️ Run
```bash
python qabot.py
```
The app starts at **http://0.0.0.0:7860**. If `share=True` is enabled, you’ll also get a temporary public URL.

---

## 🧪 How it Works (High‑Level)
1. **Load PDF** with `PyPDFLoader` → produce `Document` objects.  
2. **Chunk** with `RecursiveCharacterTextSplitter` (1k chars, 150 overlap).  
3. **Embed** chunks with `sentence-transformers/all-MiniLM-L6-v2`.  
4. **Index** in **Chroma** and retrieve top‑k relevant chunks.  
5. **Compress** with an LLM chain extractor to prune irrelevant text.  
6. **Answer** via Gemini using a strict, grounded prompt.

---

## 📝 Prompt Contract (excerpt)
- Use **only** the provided document.
- If unknown, reply: *“I don't know based on the document.”*
- Prefer structured, detailed answers with brief quotes/page refs when possible.

---

## 🚀 Roadmap
- Multi‑PDF ingestion
- Persistent Chroma store (on-disk)
- Streaming answers
- Deploy to Hugging Face Spaces / Render
- Add citations UI (expandable source snippets)

---

## 📜 License
© 2025 **Ali Maqsood** — All rights reserved.
