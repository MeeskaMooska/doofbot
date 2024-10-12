from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load and process the PDF
local_path = "RISC-V Assembly Language Primer for ESP32-C3.pdf"
loader = UnstructuredPDFLoader(file_path=local_path)
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="doofenshmirtz-llama3-model")
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create LLM
llm = ChatOllama(model="doofenshmirtz-llama3-model")

# Create prompt template
template = """You are an AI assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer: """

prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat loop
print("RAG Chatbot initialized. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    response = rag_chain.invoke(user_input)
    print("Chatbot:", response)