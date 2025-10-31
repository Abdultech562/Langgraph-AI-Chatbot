from db import users_col
import bcrypt

def register_user(username, password):
 if users_col.find_one({"username": username}):
   return "Username already exists!"
 hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
 users_col.insert_one({
 "username": username,
 "password": hashed,
 "name": None,
 "sessions": []
})
 return "User registered successfully!"

def login_user(username, password):
 user = users_col.find_one({"username": username})
 if user and bcrypt.checkpw(password.encode(), user["password"]):
  return True
 return False