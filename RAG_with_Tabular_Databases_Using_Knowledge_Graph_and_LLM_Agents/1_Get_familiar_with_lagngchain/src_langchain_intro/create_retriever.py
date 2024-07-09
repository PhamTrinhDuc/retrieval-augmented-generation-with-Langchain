import dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

def create_retriever():

    # load data
    loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
    reviews = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                          chunk_overlap=100)
    chunks = splitter.split_documents(reviews)

    # init embedding model
    embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


    # initialize the bm25 retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2
    
    #initialize the chroma retriever
    chroma_vectorstore = Chroma.from_documents(chunks, embedding, persist_directory=REVIEWS_CHROMA_PATH)
    chroma_retrieval = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

    # fusion Chroma and BM25
    ensemble_retriever = EnsembleRetriever(retrievers=[chroma_retrieval, bm25_retriever],
                                           weights=[0.5, 0.5])
    
    # rerank with cohere
    compressor = CohereRerank(cohere_api_key="SKDNVMyQRo7ZlAyziWOIpPYYrG9E7XzDMVPmQtH5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    return ensemble_retriever, embedding

if __name__ == "__main__":

    question = """Has anyone complained about
    communication with the hospital staff?"""

    ensemble_retriever, embedding = create_retriever()
    results = ensemble_retriever.get_relevant_documents(question)
    print(results[0].page_content)
    print("=" * 50)
    print(results[1].page_content)
