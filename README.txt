# AI Chatbot App

## Description

AI Chatbot App is an intelligent conversational assistant built using **Python**, **LangGraph**, and **Retrieval-Augmented Generation (RAG)** architecture.  
It allows users to chat with an AI model (Gemini 2.0 Flash), store chat history in **MongoDB**, and interact through a user-friendly **Gradio interface**.  
The chatbot can remember user sessions, retrieve relevant information from a local knowledge base, and provide **context-aware, source-backed answers**.


## Features
-> User authentication (login & register)  
-> Individual chat history per user (stored in MongoDB)  
-> **RAG-based contextual retrieval using Chroma  
-> Real-time conversation using LangGraph + Gemini 2.0 Flash model  
-> Context-aware replies with session memory  
-> Gradio-based interactive chat interface  
-> Secure password hashing (bcrypt)  
-> Env-based API key management  
-> **Automatic document embedding and retrieval**  


## Installation
-> Go to Python official website 
-> Search and install python 3.13
-> Then install requirements.txt using below command
cmd: pip install -r Requirements.txt
-> Then go to Mongodb official website
-> Search and install mongodb community edition
-> After installation install mongodb driver Pymongo for accessing mongo database
-> Redirect to python directory then install pymongo using bellow command 
cmd: python -m pip install pymongo 4.15.3



