from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

import streamlit  as st

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )


parser = StrOutputParser()

template = PromptTemplate(
    template = 'Generate 5 intresting facts about {topic}',
    input_variables = ['topic']
)

chain = template | model | parser

result = chain.invoke({'topic': 'UPSC exam'})

print(result)

chain.get_graph().print_ascii()
