# rag-multi-pdf-chatbot

This is a Multi-RAG Gradio-based web application to read, process, and interact with PDFs through a conversational AI chatbot.

## Features
- Upload PDFs
- Ask questions related to PDF content
- Uses Langchain and FAISS for vector search

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. Create and activate a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install all the dependencies
   ```bash
   pip install -r requirements.txt

4. Setup your environment variables
   ```bash
   cp .env.example .env

  Replace your_api_key_here with your actual Groq API key.

5. Run the application
     ```bash
     python app.py
