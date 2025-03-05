import gradio as gr
import logging
import os
import spacy
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
FAISS_DB_PATH = "faiss_db"


# At the top of your file, after imports
def setup_spacy():
    try:
        spacy.load("en_core_web_sm")
        logger.info("Spacy model already installed")
    except OSError:
        logger.info("Downloading spacy model...")
        os.system("python -m spacy download en_core_web_sm")
        logger.info("Spacy model downloaded successfully")

# Call the setup function
setup_spacy()

# Load environment variables from .env file
load_dotenv()

# Validate API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Set up embeddings using SpaCy
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def pdf_read(pdf_doc):
    try:
        text = ""
        for pdf in pdf_doc:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise Exception("Error processing PDF file. Please check the file format.")

def get_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        raise Exception("Error processing text chunks")

def vector_store(text_chunks):
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        os.makedirs(FAISS_DB_PATH, exist_ok=True)
        vector_store.save_local(FAISS_DB_PATH)
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise Exception("Error creating vector database")

def get_conversational_chain(tools, ques):
    try:
        # Initialize Groq LLM with API key from environment variable
        llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0  
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the context." Don't provide a wrong answer.""",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
        tool = [tools]
        # Create the agent using create_tool_calling_agent
        agent = create_tool_calling_agent(llm, tool, prompt)

        # Create an AgentExecutor to run the agent with tools
        agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
        response = agent_executor.invoke({"input": ques})
        
        return response['output']
    except Exception as e:
        logger.error(f"Error in conversation chain: {str(e)}")
        return "An error occurred while processing your request. Please try again."

def user_input(user_question):
    try:
        new_db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()
        retrieval_chain = create_retriever_tool(
            retriever, 
            "pdf_extractor", 
            "This tool is to give answers to queries from the PDF."
        )
        return get_conversational_chain(retrieval_chain, user_question)
    except Exception as e:
        logger.error(f"Error in user input processing: {str(e)}")
        return "Error processing your question. Please ensure PDFs are uploaded and processed first."

def process_pdfs(pdf_docs):
    try:
        raw_text = pdf_read(pdf_docs)
        text_chunks = get_chunks(raw_text)
        vector_store(text_chunks)
        return "PDFs processed successfully!"
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        return f"Error processing PDFs: {str(e)}"

# Health check endpoint
def healthcheck():
    return {"status": "healthy"}

# Gradio Interface Setup
with gr.Blocks() as demo:
    gr.Markdown("# RAG based Chat with PDF")
    
    with gr.Row():
        pdf_upload = gr.File(
            label="Upload your PDF Files", 
            file_count="multiple",
            file_types=[".pdf"]
        )
        submit_button = gr.Button("Submit & Process")
    
    output_message = gr.Textbox(label="Processing Status")
    
    user_question = gr.Textbox(
        label="Ask a Question from the PDF Files",
        placeholder="Type your question here..."
    )
    
    response_output = gr.Textbox(label="Reply")

    def handle_submit(pdf_docs, question):
        try:
            if not pdf_docs:
                return "Please upload PDF files first", "No PDFs processed"
            if not question:
                return "PDFs processed successfully!", "Please ask a question"
            
            process_status = process_pdfs(pdf_docs)
            answer = user_input(question)
            return process_status, answer
        except Exception as e:
            logger.error(f"Error in submission: {str(e)}")
            return f"Error processing request: {str(e)}", "Please try again"

    submit_button.click(
        handle_submit,
        inputs=[pdf_upload, user_question],
        outputs=[output_message, response_output]
    )
    
    # Allow pressing Enter to submit the question as well
    user_question.submit(
        handle_submit,
        inputs=[pdf_upload, user_question],
        outputs=[output_message, response_output]
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
