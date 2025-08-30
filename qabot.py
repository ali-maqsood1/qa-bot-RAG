import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# --- LLM (Gemini) ---
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",       # LLM model
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,             # factual
        max_output_tokens=4096       # smaller = faster
    )

# --- PDF Loader ---
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# --- Text Splitter ---
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_documents(data)

# --- Embedding model ---
def embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Chroma Database (persistent) ---
PERSIST_DIR = "chroma_storage"
retriever_cache = {}

def build_or_load_vectorstore(file_path):
    """Build a new Chroma DB for this PDF if not cached, else load from disk."""
    embeddings = embedding_model()
    pdf_name = os.path.splitext(os.path.basename(file_path))[0]
    pdf_db_dir = os.path.join(PERSIST_DIR, pdf_name)

    # If DB already exists, load it
    if os.path.exists(pdf_db_dir):
        vectordb = Chroma(persist_directory=pdf_db_dir, embedding_function=embeddings)
    else:
        os.makedirs(pdf_db_dir, exist_ok=True)
        docs = document_loader(file_path)
        chunks = text_splitter(docs)
        vectordb = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=pdf_db_dir
        )
        vectordb.persist()
    return vectordb

def get_retriever(file_path):
    """Cache the retriever for this PDF to avoid rebuilding every query."""
    if file_path not in retriever_cache:
        vectordb = build_or_load_vectorstore(file_path)
        retriever_cache[file_path] = vectordb.as_retriever(search_kwargs={"k":8})
    return retriever_cache[file_path]

# --- Prompt ---
prompt_template = """
You are an expert assistant. Use ONLY the content from the provided document to answer.
- Give structured, detailed answers.
- If information is missing, say: "I don't know based on the document."
- Prefer clarity over brevity. Avoid vague statements.
- Where possible, provide brief supporting quotes or page references from the document.

Document context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --- Retrieval QA ---
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = get_retriever(file.name)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    response = qa.invoke({"query": query})
    return response['result']

# --- Custom CSS ---
theme_css = """
.gradio-container {
    background-color: #f7fbff;
    font-family: 'Segoe UI', Tahoma, sans-serif;
}
header h1 {
    text-align: center;
    background: linear-gradient(90deg, #004aad, #007bff);
    color: white;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
    font-size: 28px;
}
.gradio-interface .input-text textarea,
.gradio-interface .output-text textarea {
    border-radius: 12px;
    border: 2px solid #004aad;
    padding: 10px;
    font-size: 16px;
}
footer {
    text-align: center;
    color: #004aad;
    font-size: 14px;
    padding: 15px;
    margin-top: 20px;
    border-top: 1px solid #004aad;
    background-color: #eaf3ff;
}
"""

# --- Wrapper for Footer ---
def app_with_footer(file, query):
    return retriever_qa(file, query)

# --- Gradio UI ---
with gr.Blocks(css=theme_css) as rag_application:
    gr.HTML("<header><h1>📘 PDF Question-Answering Bot</h1></header>")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF File", file_types=['.pdf'], type="filepath")
            query_input = gr.Textbox(label="Ask a Question", lines=2, placeholder="Type your question...")
            submit_btn = gr.Button("Get Answer", variant="primary")
        with gr.Column():
            answer_output = gr.Textbox(label="Answer", lines=12)
    
    submit_btn.click(fn=app_with_footer, inputs=[file_input, query_input], outputs=[answer_output])
    
    gr.HTML("<footer>©2025 Ali Maqsood — All Rights Reserved</footer>")

if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860, share=True)
