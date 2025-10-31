from dotenv import load_dotenv
import os


load_dotenv()


GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
NESTLE_PDF_PATH = "NESTLE-financial-statements-2024.pdf"
CHROMA_PATH = "chroma_db"


if not GEMINI_KEY:
 raise ValueError("GEMINI_API_KEY not found in .env file!")
if not MONGO_URI:
 raise ValueError("MONGO_URI not found in .env file!")