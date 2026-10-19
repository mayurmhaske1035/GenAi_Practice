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
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )

class facts(BaseModel):
    fact_1 : str = Field(description='Fact 1 about topic')
    fact_2 : str = Field(description='Fact 2 about topic')
    fact_3 : str = Field(description='Fact 3 about topic')

parser = PydanticOutputParser(pydantic_object=facts)

template = PromptTemplate(
    template = (
        'Give me exactly 3 facts about {topic}. \n'
        'Return only a valid JSON object with the foloowing keys: \n'
        "fact_1, fact_2, fact_3"
        'Do not include schema, properties, descriptions, or explanations. \n'
        'Only return the valid JSON \n'
        '{format_instructions}'
    ),
    input_variables = ['topic'],
    partial_variables = {"format_instructions": parser.get_format_instructions()}

)

prompt = template.invoke({'topic':'MIT Research Boston'})

# print(prompt)

result = model.invoke(prompt)

print(result)

final_result = parser.parse(result.content[0]["text"])

print("final_result")
print(final_result)