import os
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import joblib


# Helper function to load and parse the input data
def load_or_parse_data():
    data_file = "data/parsed_data.pkl"

    if os.path.exists(data_file):
        parsed_data = joblib.load(data_file)
    else:
        parsing_instruction = """
        Tài liệu được cung cấp là thông tin các sản phẩm của chúng tôi.
        Tài liệu gồm 2 phần: Các loại sản phẩm và thông tin sản phẩm.
        Mỗi sản phẩm chứa các thông tin như ID, tên sản phẩm, giá bán, mô tả sản phẩm, số lượng...
        """

        parser = LlamaParse(api_key=os.getenv("LLAMAPARSE_API_KEY"),
                            result_type="markdown",
                            parsing_instruction=parsing_instruction,
                            max_timeout=5000,)
        llama_parse_documents = parser.load_data("data/product.xlsx")


        # Save the parsed data to a file
        # print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data

# Helper function to load chunks into vectorstore.

def create_vector_database():

    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data()

    if not os.path.exists("data/output.md"):
        with open('data/output.md', 'a') as f:  # Open the file in append mode ('a')
            for doc in llama_parse_documents:
                f.write(doc.text + '\n')
        
    markdown_path = "data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

    loader = DirectoryLoader("data/", "**/*.md", show_progress=True)
    documents = loader.load()

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    docs = text_splitter.split_documents(documents=documents)

    # Initialize Embeddings
    embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


    # initialize the bm25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2
    
    #initialize the chroma retriever
    faiss_vectorstore = FAISS.from_documents(docs, embedding)
    faiss_retrieval = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

    # fusion FAISS and BM25
    ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retrieval, bm25_retriever],
                                           weights=[0.5, 0.5])
    
    # rerank with cohere
    compressor = CohereRerank(cohere_api_key="SKDNVMyQRo7ZlAyziWOIpPYYrG9E7XzDMVPmQtH5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )

    return compression_retriever, embedding
if __name__ == "__main__":
    compression_retriever, embed_model = create_vector_database()



# def set_custom_prompt(self):
#     _template = """Đưa ra cuộc trò chuyện sau đây và một câu hỏi tiếp theo, hãy diễn đạt lại câu hỏi tiếp theo thành một câu hỏi độc lập. 
#     Bạn có thể đặt câu hỏi về trạng thái gần đây nhất của địa chỉ công đoàn.
#     Lưu ý: bạn chỉ được sử dụng tiếng việt để trả lời.

#     Chat History:
#     {chat_history}
#     Follow Up Input: {question}
#     Standalone question:"""
#     CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

#     template = """Bạn là trợ lý AI để trả lời các câu hỏi về  các thông tin của sản phẩm. 
#     Bạn được cung cấp các phần được trích xuất sau đây của một tài liệu dài và một câu hỏi. Đưa ra câu trả lời đàm thoại.
#     Nếu bạn không biết câu trả lời, chỉ cần nói "Hmm, tôi không chắc." 
#     Đừng cố gắng bịa ra một câu trả lời. Nếu câu hỏi không phải về  sản phẩm, hãy sử dụng tri thức của bạn để trả lời. 
#     Cuối cùng, hãy trả lời câu hỏi như thể bạn là một tên cướp biển đến từ vùng biển phía Nam và vừa trở về sau chuyến thám hiểm cướp biển, nơi bạn tìm thấy một rương kho báu chứa đầy tiền vàng.
#     Lưu ý: bạn chỉ được sử dụng tiếng việt để trả lời.
    
#     Question: {question}
#     =========
#     {context}
#     =========
#     Answer in Markdown:"""
#     QA_PROMPT = PromptTemplate(template=template, input_variables=[
#                             "question", "context"])

#     return CONDENSE_QUESTION_PROMPT, QA_PROMPT