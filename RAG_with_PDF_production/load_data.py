from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dotenv
import os



DATA_DIR = "data/pdf_product.pdf"
VECTOR_DB = "vector_db"
dotenv.load_dotenv()
# Initialize Embeddings
embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def load_data():
    documents = None

    if DATA_DIR.endswith("xlsx"):
        loader = UnstructuredExcelLoader(DATA_DIR, mode="elements")
        documents = loader.load()

    else:
        loader = PDFPlumberLoader(DATA_DIR)
        documents = loader.load()

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    docs = text_splitter.split_documents(documents=documents)

    # initialize the bm25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2
    
    #initialize the chroma retriever
    if not os.path.exists(VECTOR_DB):
        faiss_vectorstore = FAISS.from_documents(docs, embedding)
        faiss_vectorstore.save_local(VECTOR_DB)
    else:
        faiss_vectorstore = FAISS.load_local(VECTOR_DB, embedding, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2, "similarity_score": 0.6})

    # fusion FAISS and BM25
    ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever],
                                           weights=[0.5, 0.5])
    
    # rerank with cohere
    compressor = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"))
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    return compression_retriever, embedding
if __name__ == "__main__":
    compression_retriever = load_data()

