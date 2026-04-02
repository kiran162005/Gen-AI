from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Step 1: Load Document (Parsing Stage)
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# Step 2: Parse / Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Step 3: Convert Text into Embeddings
embeddings = OpenAIEmbeddings(openai_api_key="YOUR_API_KEY")

# Step 4: Store in Vector Database (FAISS)
vector_store = FAISS.from_documents(docs, embeddings)

# Step 5: Create Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 6: Create RAG Pipeline
llm = OpenAI(openai_api_key="YOUR_API_KEY", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Step 7: Ask Query
query = "What is the main topic of the document?"
response = qa_chain.run(query)

print("Answer:", response)
