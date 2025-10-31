import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
from config import NESTLE_PDF_PATH


# Load and chunk the Nestle PDF
def load_and_chunk_pdf(chunk_size=1000, chunk_overlap=200):
    if not os.path.exists(NESTLE_PDF_PATH):
        raise FileNotFoundError(
            f"❌ {NESTLE_PDF_PATH} not found! Please place the file in the correct folder."
        )

    print("🔍 Loading PDF using PyMuPDFLoader...")
    loader = PyMuPDFLoader(NESTLE_PDF_PATH)
    documents_raw = loader.load()
    print(f"✅ PyMuPDFLoader: successfully loaded {len(documents_raw)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []
    for i, doc in enumerate(documents_raw):
        page_content = doc.page_content.strip()
        if not page_content:
            continue
        split_texts = text_splitter.split_text(page_content)
        for chunk_text in split_texts:
            chunks.append(Document(
                page_content=chunk_text,
                metadata={"page": i + 1, "section": "Financial Statement"}
            ))

    print(f"✅ Created {len(chunks)} text chunks from Nestlé financial report.")

    # Build keyword -> pages index (financial keywords)
    keyword_page_index = {}
    for doc in chunks:
        page = doc.metadata.get("page")
        text = doc.page_content.lower()
        keywords = re.findall(
            r'\b(net profit|total revenue|equity|cash|income|assets|liabilities|shareholders)\b', text
        )
        for kw in keywords:
            if kw not in keyword_page_index:
                keyword_page_index[kw] = set()
            keyword_page_index[kw].add(page)

    print("✅ Keyword-to-page index created.")
    return chunks, keyword_page_index
