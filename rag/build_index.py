from dotenv import load_dotenv
import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

# Charger la clé API depuis .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configurer Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

# 📄 Liste des fichiers
doc_files = [
    "data/Questions & réponses WinCompta.docx",
    "data/_Questions & réponses WinHub.docx",
    "data/Guide WinHub (1).pdf"   
]

docs = []

for file_path in doc_files:
    if os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == ".pdf":
            loader = PyMuPDFLoader(file_path)
        else:
            print(f"⚠️ Format non supporté : {file_path}")
            continue
        
        docs.extend(loader.load())
        print(f"✅ Chargé : {file_path}")
    else:
        print(f"⚠️ Fichier non trouvé : {file_path}")

# ✂️ Splitter les documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splitted_docs = text_splitter.split_documents(docs)

# 🔍 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 📦 Indexer FAISS
db = FAISS.from_documents(splitted_docs, embeddings)

# 💾 Sauvegarder l'index
output_dir = "rag/faiss_index"
os.makedirs(output_dir, exist_ok=True)
db.save_local(output_dir)

print("✅ Index FAISS généré et sauvegardé avec succès.")













