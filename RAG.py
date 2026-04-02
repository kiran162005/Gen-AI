# full_rag_pipeline.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# -----------------------------
# Step 0: Set your OpenAI API key
# -----------------------------
OPENAI_API_KEY = "YOUR_API_KEY"

# -----------------------------
# Step 1: Load PDF document
# -----------------------------
loader = PyPDFLoader("sample.pdf")   # replace with your PDF path
documents = loader.load()

# -----------------------------
# Step 2: Split text into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # size of each chunk
    chunk_overlap=50      # overlap between chunks
)
docs = text_splitter.split_documents(documents)

# -----------------------------
# Step 3: Create embeddings
# -----------------------------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# -----------------------------
# Step 4: Store embeddings in FAISS
# -----------------------------
vector_store = FAISS.from_documents(docs, embeddings)

# -----------------------------
# Step 5: Create a retriever
# -----------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # top 3 chunks

# -----------------------------
# Step 6: Initialize LLM
# -----------------------------
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# -----------------------------
# Step 7: Create RetrievalQA chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"    # simple chain that stuffs chunks into the LLM
)

# -----------------------------
# Step 8: Ask a query
# -----------------------------
query = "What is the main topic of the document?"

# Use modern invoke() method for compatibility
response = qa_chain.invoke({"query": query})

print("Answer:", response["result"])