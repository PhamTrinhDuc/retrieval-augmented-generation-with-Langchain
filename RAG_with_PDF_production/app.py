import streamlit as st
from llm import ChatBot


bot = ChatBot()
chain = bot.get_condense_prompt_qa_chain()

def respond(query: str) -> str:
    response = chain.invoke({"question": query})
    return response['answer']

# deploy model using streamlit
st.set_page_config(page_title="Langchain RAG Chatbot", page_icon=":robot_face:")


with st.sidebar:
    st.header("About")
    st.markdown(
        """
        Đây là 1 giao diện chatbot
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        được thiết kế để hỏi đáp, tư vấn người dùng về các thông tin của sản phẩm.
        """
    )

    st.header("Các câu hỏi mẫu bạn có thể hỏi")
    st.markdown("- Bên bạn có bán nồi cơm điện không ?")
    st.markdown("- Tôi muốn mua điều hòa giá 10 triệu.")
    st.markdown(
        "- Tôi muốn mua nồi cơm điện dành cho gia đình 4 người."
    )
    st.markdown("- Hãy cung cấp cho tôi sản phẩm máy ép có giá rẻ nhất.")
    st.markdown(
        "- Nồi chiên không dầu bên bạn bảo hành trong bao lâu ?"
    )
    st.markdown("- Đèn năng lượng mặt trời có dung tích 2 lít")
    st.markdown("- Đèn năng lượng mặt trời có công suất cao")
    st.markdown("- Máy giạt nào có giá dưới 20 triệu ?")
    st.markdown("- Nồi chiên không dầu 15 lít có giá dưới 20 triệu")


st.markdown("""
<div style="text-align: center;">
            <img src="https://blogs.perficient.com/files/lanchain.png" alt="Chatbot Logo" width="100"/>
    <img src="https://img.freepik.com/premium-vector/robot-icon-chat-bot-sign-support-service-concept-chatbot-character-flat-style_41737-796.jpg?" alt="Chatbot Logo" width="200"/>
    <h1 style="color: #0078D7;">Langchain based RAG Chatbot</h1>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<p style="text-align: center; font-size: 18px; color: #555;">
    Xin chào !! Tôi là 1 chatbot hỗ trợ khách hàng.
</p>
""", unsafe_allow_html=True)


st.markdown("<hr/>", unsafe_allow_html=True)

user_query = st.text_input("Enter your question:", placeholder="E.g., What is the aim of AI act?")

if st.button("Answer"):
    bot_response = respond(user_query)
   
    st.markdown(f"""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-top: 20px;">
        <h4 style="color: #0078D7;">Bot's Response:</h4>
        <p style="color: #335;">{bot_response}</p>
    </div>
    """, unsafe_allow_html=True)