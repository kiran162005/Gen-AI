import os
import sys
import requests
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# -----------------------------
# Configuration
# -----------------------------
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
IPC_URL = "https://www.mha.gov.in/sites/default/files/IPAct_1860.pdf"
IPC_FILE = "Indian_Penal_Code.pdf"

# -----------------------------
# Step 1: Download PDF
# -----------------------------
def download_pdf(url, filename):
    print("Downloading Indian Penal Code document...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete.")
    except requests.RequestException as e:
        print(f"Error downloading PDF: {e}")
        sys.exit(1)

if not os.path.exists(IPC_FILE):
    download_pdf(IPC_URL, IPC_FILE)

# -----------------------------
# Step 2: Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        sys.exit(1)

print("Extracting text from the IPC document...")
ipc_text = extract_text_from_pdf(IPC_FILE)

if not ipc_text:
    print("Error: Failed to extract text from the PDF.")
    sys.exit(1)

# -----------------------------
# Step 3: Split text into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
ipc_chunks = text_splitter.split_text(ipc_text)

# -----------------------------
# Step 4: Create FAISS vector store
# -----------------------------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = FAISS.from_texts(ipc_chunks, embedding=embeddings)

# -----------------------------
# Step 5: Load Chat Model
# -----------------------------
chat_model = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# -----------------------------
# Step 6: Create Retrieval QA chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

# -----------------------------
# Step 7: Chatbot interface
# -----------------------------
def ipc_chatbot():
    print("\nIndian Penal Code Chatbot (Type 'exit' to stop)")
    while True:
        query = input("\nAsk about the Indian Penal Code: ").strip()
        if query.lower() == "exit":
            print("Exiting chatbot. Have a great day!")
            sys.exit(0)
        response = qa_chain.run(query)
        print("\nIPC Chatbot: ", response)

if __name__ == "__main__":
    ipc_chatbot()