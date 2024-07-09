from langchain_groq import ChatGroq
from langchain.prompts import (
    PromptTemplate,
)
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
import dotenv
import os
from load_data import load_data

dotenv.load_dotenv()

class ChatBot:
    def __init__(self):
        self.CONDENSE_QUESTION_PROMPT, self.QA_PROMPT = self.set_custom_prompt()
        self.compression_retriever, self.embed_model = load_data()
        self.LLM = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            verbose=True,
            max_tokens=850
        )
        self.memory = self.create_memory()
    

    def create_memory(self):
        embedding_size = 384 # dimension of the embedding model
        index =faiss.IndexFlatL2(embedding_size)
        vector_store = FAISS(self.embed_model.embed_query, 
                            index,
                            InMemoryDocstore({}), {})

        # Create VectorStoreRetrieverMemory
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="chat_history", input_key="question")

        return memory

    def set_custom_prompt(self):
        _template = """Đưa ra cuộc trò chuyện sau đây và một câu hỏi tiếp theo, hãy diễn đạt lại câu hỏi tiếp theo thành một câu hỏi độc lập. 
        Bạn có thể đặt câu hỏi về trạng thái gần đây nhất của địa chỉ công đoàn.
        Lưu ý: bạn chỉ được sử dụng tiếng việt để trả lời.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        template = """Bạn là trợ lý AI để trả lời các câu hỏi về  các thông tin của sản phẩm. 
        Bạn được cung cấp các phần được trích xuất sau đây của một tài liệu dài và một câu hỏi. Đưa ra câu trả lời đàm thoại.
        Nếu bạn không biết câu trả lời, chỉ cần nói "Hmm, tôi không chắc." 
        Đừng cố gắng bịa ra một câu trả lời. Nếu câu hỏi không phải về  sản phẩm, lúc này hãy sử dụng tri thức của bạn để trả lời.
        Lưu ý: bạn chỉ được sử dụng tiếng việt để trả lời.
        
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown:"""
        QA_PROMPT = PromptTemplate(template=template, input_variables=[
                                "question", "context"])

        return CONDENSE_QUESTION_PROMPT, QA_PROMPT


    # Instantiate the Retrieval Question Answering Chain
    def get_condense_prompt_qa_chain(self):

        model = ConversationalRetrievalChain.from_llm(
            llm=self.LLM,
            retriever=self.compression_retriever,
            memory=self.memory,
            condense_question_prompt=self.CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": self.QA_PROMPT})
        
        return model

if __name__ == "__main__":
    bot = ChatBot()

    chain = bot.get_condense_prompt_qa_chain()
    response = chain.invoke({"question": "bên bạn có sản phẩm tủ lạnh toshiba không ?"})
    print(response['answer'])

