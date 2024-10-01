import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Importing Groq LLM
from langchain.agents import create_tool_calling_agent, AgentExecutor  # Importing agent functions
import os

# Load environment variables from .env file
load_dotenv()


# Set up embeddings using SpaCy
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")



def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conversational_chain(tools, ques):
    # Initialize Groq LLM with API key from environment variable
    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY"),  # Ensure this is set correctly
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

def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answers to queries from the PDF.")
    
    return get_conversational_chain(retrieval_chain, user_question)

def process_pdfs(pdf_docs):
    raw_text = pdf_read(pdf_docs)
    text_chunks = get_chunks(raw_text)
    vector_store(text_chunks)
    return "PDFs processed successfully!"

# Gradio Interface Setup
with gr.Blocks() as demo:
    gr.Markdown("# RAG based Chat with PDF")
    
    with gr.Row():
        pdf_upload = gr.File(label="Upload your PDF Files", file_count="multiple")
        submit_button = gr.Button("Submit & Process")
    
    output_message = gr.Textbox(label="Processing Status")
    
    user_question = gr.Textbox(label="Ask a Question from the PDF Files", placeholder="Type your question here...")
    
    response_output = gr.Textbox(label="Reply")

    def handle_submit(pdf_docs, question):
        process_status = process_pdfs(pdf_docs)
        answer = user_input(question)
        return process_status, answer

    submit_button.click(handle_submit, inputs=[pdf_upload, user_question], outputs=[output_message, response_output])
    
    # Allow pressing Enter to submit the question as well
    user_question.submit(handle_submit, inputs=[pdf_upload, user_question], outputs=[output_message, response_output])

# Launch the Gradio interface
demo.launch()
