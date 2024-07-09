import gradio as gr
from llm import ChatBot

bot = ChatBot()

# Function to handle chat messages
def chat(input_text, history=[]):
    inputs = {"input": input_text}

    response = bot.chain.invoke(input_text)
    # Save the conversation in memory
    bot.memory.save_context(inputs, {"output": response['result']})
    
    # Append the new interaction to the history
    history.append((input_text, response['result']))
    return response['result'], history

# Create Gradio interface
def run_app():
    with gr.Blocks() as iface:
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Gõ tin nhắn")
        clear = gr.Button("Clear")

        def user_interaction(user_message, history):
            bot_response, updated_history = bot.chat(user_message, history)
            return updated_history, updated_history

        msg.submit(user_interaction, [msg, chatbot], [chatbot, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    # Launch the interface
    iface.launch()

if __name__ == "__main__":
    run_app()
