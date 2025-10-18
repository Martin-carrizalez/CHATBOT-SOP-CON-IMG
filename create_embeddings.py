"""
Crea embeddings con Hugging Face (gratis, sin límites)
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "guia_sop.pdf"

print("📖 Cargando PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

for doc in documents:
    doc.metadata['fuente'] = 'Guía ESHRE 2023'

print(f"✅ {len(documents)} páginas")

print("✂️ Dividiendo...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)

print(f"✅ {len(chunks)} chunks")

print("🧠 Creando embeddings con Hugging Face...")
print("   (Primera vez descarga modelo ~500MB - puede tardar)")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("💾 Creando vectorstore...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_sop",
    collection_name="sop_medical_guide"
)

print("✅ ¡LISTO! Vectorstore guardado en ./chroma_db_sop")
print("\nAhora ejecuta: streamlit run bot_sop.py")