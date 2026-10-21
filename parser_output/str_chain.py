from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit  as st
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Optional, Literal
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )


template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)


template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | template2 | model | parser

result = chain.invoke({'topic': 'india'})

print(result)