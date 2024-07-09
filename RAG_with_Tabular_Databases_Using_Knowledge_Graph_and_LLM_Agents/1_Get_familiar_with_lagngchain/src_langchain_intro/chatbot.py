import dotenv
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from tools import get_current_wait_time
from create_retriever import create_retriever
from langchain_groq import ChatGroq


REVIEWS_CHROMA_PATH = "chroma_data"
dotenv.load_dotenv()

class Chatbot:

    def __init__(self):
        self.review_prompt_template = self.create_prompt()
        self.ensemble_retriever, self.embedding = create_retriever()
        self.chat_model = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.0
        )

    # create prompt template
    def create_prompt(self):
        review_template_str = """Your job is to use patient
        reviews to answer questions about their experience at
        a hospital. Use the following context to answer questions.
        Be as detailed as possible, but don't make up any information
        that's not from the context. If you don't know an answer, say
        you don't know.

        {context}
        """
        review_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context"],
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

        review_prompt_template = ChatPromptTemplate(
            input_variables=["context", "question"],
            messages=messages,
)

        return review_prompt_template

    # create chain: fusion between llm, retrieval, prompt, output_parser
    def create_chain(self):

        review_chain = (
            {"context": self.ensemble_retriever, "question": RunnablePassthrough()}
            | self.review_prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        return review_chain


    # create tools for agent
    def create_agents(self):
        chain = self.create_chain()
        tools = [
            Tool( # Create an agent responsible for answering questions about hospital information 
                name = "Reviewer",
                func= chain.invoke,
                description="""Useful when you need to answer questions
                about patient reviews or experiences at the hospital.
                Not useful for answering questions about specific visit
                details such as payer, billing, treatment, diagnosis,
                chief complaint, hospital, or physician information.
                Pass the entire question as input to the tool. For instance,
                if the question is "What do patients think about the triage system?",
                the input should be "What do patients think about the triage system?"
                """,
            ),
            Tool( # Create an agent responsible for responding to hospital waiting times
                name="Waits" ,
                func=get_current_wait_time,
                description="""Use when asked about current wait times
                at a specific hospital. This tool can only get the current
                wait time at a hospital and does not have any information about
                aggregate or historical wait times. This tool returns wait times in
                minutes. Do not pass the word "hospital" as input,
                only the hospital name itself. For instance, if the question is
                "What is the wait time at hospital A?", the input should be "A".
                """,
            ),
        ]

        pital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")
        hospital_agent = create_openai_functions_agent(
            llm=self.chat_model,
            prompt=pital_agent_prompt,
            tools=tools
        )

        hospital_agent_executor = AgentExecutor(
            agent=hospital_agent,
            tools=tools,
            return_intermediate_steps=True,
            verbose=True
            # return_intermediate_steps and verbose to True will allow you to see the agent’s thought process and the tools it calls. 
        )

        return hospital_agent_executor
    
if __name__ == "__main__":

    chatbot_hospital = Chatbot()

    ################# TEST PROMPT ####################
    # chain_prompt = chatbot_hospital.review_prompt_template | chatbot_hospital.chat_model
    # context = "I had a great stay!"
    # question = "Did anyone have a positive experience?"

    # results = chain_prompt.invoke({"context": context, "question": question})
    # print(results.content)

    ################# TEST CHAIN #####################
    review_chain = chatbot_hospital.create_chain()
    results = review_chain.invoke(
            """who had a comfortable stay at Vaughn PLC hospital"""
    )
    print(results)

    ################# TEST AGENT #####################
    hospital_agent = chatbot_hospital.create_agents()

    # hospital_agent.invoke(
    #     {"input": "What is current time at hospital C?"}
    # )

    hospital_agent.invoke(
        {"input": """"Show the reviews written by patient 7674"""}
    )
    
