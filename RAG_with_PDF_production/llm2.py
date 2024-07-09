from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
import dotenv
import os
from load_data import load_data

dotenv.load_dotenv()
# memory = ConversationBufferMemory(return_messages=True)
# memory.load_memory_variables({})



class ChatBot:
    def __init__(self):
        self.prompt = self.set_custom_prompt()
        self.compression_retriever, self.embed_model = load_data()
        self.LLM = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            verbose=True,
            max_tokens=850
        )
        self.memory = self.create_memory()
        self.chain = self.create_Chain_QA()
    

    def create_memory(self):
        embedding_size = 384 # dimension of the embedding model
        index =faiss.IndexFlatL2(embedding_size)
        vector_store = FAISS(self.embed_model.embed_query, 
                            index,
                            InMemoryDocstore({}), {})

        # Create VectorStoreRetrieverMemory
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="history", input_key="question")

        return memory

    def set_custom_prompt(self):
        review_template_str = """
        Bạn là chuyên gia tư vấn khách hàng và rất am hiểu về các sản phẩm điện tử, gia dụng.... tại Việt Nam.
        Dựa vào thông tin được cung cấp và câu hỏi từ người dùng, hãy đưa ra câu trả lời cuối cùng.
        Nếu các thông tin trong câu hỏi không có trong phần được cung cấp trước đó, hãy nói là "Tôi không biết".
        {context}

        Ngoài thông tin cho trước bạn có thể dựa vào các thông tin trong quá khứ cuộc trò chuyệnn để trả lời:
        {history}
        Lưu ý: bạn chỉ được sử dụng tiếng việt để trả lời.
        """

        review_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context", "history"],
                template=review_template_str,
            )
        )

        review_human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["question"],
                template="{question}",
            )
        )
        messages = [review_system_prompt, review_human_prompt]

        prompt_template = ChatPromptTemplate(
            input_variables=["context", "question", "history"],
            messages=messages)
        return prompt_template


    # Instantiate the Retrieval Question Answering Chain
    def create_Chain_QA(self):

        qa = RetrievalQA.from_chain_type(
            llm=self.LLM,
            chain_type='stuff',
            retriever=self.compression_retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": self.prompt,
                "memory": self.memory
            },
        
        )
        return qa

    # Function to handle chat messages
    # def chat(self, input_text: str, history=[]):
    #     # Load the current conversation history
    #     # self.memory.load_memory_variables({})
    #     inputs = {"input": input_text}
    #     response = self.chain.invoke(input_text)
    #     # Save the conversation in memory
    #     self.memory.save_context(inputs, {"output": response['result']})
        
    #     # Append the new interaction to the history
    #     history.append((input_text, response['result']))
    #     return response['result'], history
    
    # # Create Gradio interface
    # def run_app(self):
    #     with gr.Blocks() as iface:
    #         chatbot = gr.Chatbot()
    #         msg = gr.Textbox(label="Enter messages")
    #         clear = gr.Button("Clear")

    #         def user_interaction(user_message, history):
    #             bot_response, updated_history = self.chat(user_message, history)
    #             return updated_history, updated_history

    #         msg.submit(user_interaction, [msg, chatbot], [chatbot, chatbot])
    #         clear.click(lambda: None, None, chatbot, queue=False)

    #     # Launch the interface
    #     iface.launch()

# if __name__ == "__main__":
#     bot = ChatBot()
#     bot.run_app()

#     chain = bot.create_Chain_QA()
#     response = chain.invoke("bên bạn có sản phẩm tủ lạnh toshiba không ?")
#     print(response['result'])

