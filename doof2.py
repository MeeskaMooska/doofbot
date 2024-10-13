from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import time

# Load and process the PDF (do this only once)
local_path = "IChing.pdf"
loader = UnstructuredPDFLoader(file_path=local_path)
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
chunks = text_splitter.split_documents(data)

# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="doofenshmirtz-llama3-model")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Create retriever with caching
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Create LLM with streaming
llm = ChatOllama(model="doofenshmirtz-llama3-model", streaming=True)

# Create a more concise prompt template
template = """
You are Dr. Heinz Doofenshmirtz, the slightly goofy but determined villain. Answer the question in a humorous and over-the-top style, adding your trademark villainous commentary and encouragement to be productive at Doofenshmirtz Evil Inc.

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

# Chat loop with timing
print("Optimized RAG Chatbot initialized. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    start_time = time.time()
    response = rag_chain.invoke(user_input)
    end_time = time.time()
    
    print("Chatbot:", response)
    print(f"Response time: {end_time - start_time:.2f} seconds")
