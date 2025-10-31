from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from config import GEMINI_KEY, CHROMA_PATH

def setup_embeddings_and_vectorstore(chunks):
 embeddings = GoogleGenerativeAIEmbeddings(
  model="text-embedding-004",
  google_api_key=GEMINI_KEY
)
 print("✅ Embedding model loaded successfully!")


 if os.path.exists(CHROMA_PATH):
  print("💾 Loading existing Chroma vector database...")
  vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
  print("✅ Existing Chroma vector database loaded successfully!")
 else:
  print("🧠 Creating new Chroma vector database (first-time embedding)...")
  vectorstore = Chroma.from_documents(
   documents=chunks,
   embedding=embeddings,
   persist_directory=CHROMA_PATH
)
  print("✅ Chroma vector database created and automatically saved to disk!")


 return embeddings, vectorstore