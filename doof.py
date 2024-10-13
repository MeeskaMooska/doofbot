from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import time

# Load and process the PDF (unchanged)
local_path = "RISC-V Assembly Language Primer for ESP32-C3.pdf"
loader = UnstructuredPDFLoader(file_path=local_path)
data = loader.load()

# Split the document into chunks (unchanged)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
chunks = text_splitter.split_documents(data)

# Create embeddings and vector store (unchanged)
embeddings = OllamaEmbeddings(model="doofenshmirtz-llama3-model")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
vectorstore.persist()

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Create LLM
llm = ChatOllama(model="doofenshmirtz-llama3-model")

# Create two separate prompts

# Prompt for information retrieval
retrieval_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question. Be factual and concise.

Question: {question}
Context: {context}
Summary: """)

# Prompt for character response
character_prompt = ChatPromptTemplate.from_template("""
You are Dr. Heinz Doofenshmirtz, an evil scientist helping your intern understand their homework or textbook. 
Use the following information to answer the intern's question, but do so in your unique style:
- Be humorous and over-the-top
- Add villainous commentary while explaining
- Encourage the intern to be productive at Doofenshmirtz Evil Inc.
- Reference your latest "-inator" invention if relevant

Information to use in your answer: {retrieval_result}

Intern's Question: {original_question}

Dr. Doofenshmirtz's Answer: """)

# Create the two-stage RAG chain
def rag_chain(question):
    # Stage 1: Retrieve and summarize relevant information
    retrieval_result = retriever.get_relevant_documents(question)
    summary_messages = retrieval_prompt.format_messages(question=question, context=str(retrieval_result))
    summary = llm(summary_messages)
    
    # Stage 2: Generate character-based response
    character_messages = character_prompt.format_messages(
        retrieval_result=summary.content,
        original_question=question
    )
    character_response = llm(character_messages)
    
    return character_response.content
  

# Chat loop
print("Dr. Doofenshmirtz's Evil Chatbot initialized. Type 'exit' to end the conversation.")
while True:
    user_input = input("Intern: ")
    if user_input.lower() == 'exit':
        print("Dr. Doofenshmirtz: Curse you, Perry the Platypus! ...Oh, wait, wrong exit line. Goodbye, intern!")
        break
    
    start_time = time.time()
    response = rag_chain(user_input)
    end_time = time.time()
    
    print("Dr. Doofenshmirtz:", response)
    print(f"Evil response time: {end_time - start_time:.2f} seconds")