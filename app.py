import gradio as gr
import re
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# 1. API key 
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# 2. Document Loader with PyPDF
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    # Cleaning weird spacing in the PDF text
    for doc in loaded_document:
        doc.page_content = re.sub(r'(?<=[a-zA-Z])\s+(?=\s)', '', doc.page_content)
    return loaded_document

# 3. Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    return chunks

# 4. Vector store
def vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding_model)
    return vectordb

# 5. Main chat function
def answer_question(file, question):
    try:
        if not file: 
            return "Please upload a PDF first."
        if not HF_TOKEN:
            return "Error: HF_TOKEN secret is missing."

        # Process the PDF using the functions defined above
        pdf_doc = document_loader(file)
        chunks = text_splitter(pdf_doc)
        vectordb = vector_database(chunks)
        
        # Retrieve context
        relevant_docs = vectordb.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in relevant_docs])

        # Chat Completion with Llama-3
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are an assistant. Answer based on the context provided."},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
            ],
            max_tokens=400,
            temperature=0.2
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Technical Error: {str(e)}"

# display README  underneath my chatbot 
with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()
# avoid  YAML metadata (entre --- ---)
if readme.startswith("---"):
    readme = readme.split("---", 2)[-1]


# 6. Create gradio interface
rag_application = gr.Interface(
   fn=answer_question,
    inputs=[
        gr.File(label="1. Upload PDF"), 
        gr.Textbox(label="2. Ask a Question")
    ],
    outputs="text",
    title="PDF Chatbot (RAG)",
    description=readme
)

rag_application.launch()
