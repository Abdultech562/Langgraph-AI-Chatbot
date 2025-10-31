from pymongo import MongoClient
from config import MONGO_URI


client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
users_col = db["users"]