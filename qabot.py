import os
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import gradio as gr

# --- Suppress warnings ---
import warnings
warnings.filterwarnings('ignore')

# --- LLM (Gemini) ---
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # GEMMA-27B-IT # EPOCH
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,             # factual over creative
        max_output_tokens=1024        # allow longer, detailed answers
    )

# --- PDF Loader ---
def document_loader(file):
    loader = PyPDFLoader(file.name)
    return loader.load()

# --- Text Splitter ---
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    return splitter.split_documents(data)

# --- Embedding model ---
def embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Vector Database ---
def vector_database(chunks):
    embeddings = embedding_model()
    return Chroma.from_documents(chunks, embeddings)

# --- Retriever with compression ---
def retriever(file):
    docs = document_loader(file)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks)
    base_retriever = vectordb.as_retriever(search_kwargs={"k":5})
    
    # Compress irrelevant parts before passing to LLM
    compressor = LLMChainExtractor.from_llm(get_llm())
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return compressed_retriever

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
    retriever_obj = retriever(file)
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