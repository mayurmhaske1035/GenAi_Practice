from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
import streamlit  as st

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )


class Feedback(BaseModel):
    sentiment :  Literal['positive', 'negative'] = Field( description= 'Give sentiment of the feedback')

parser = PydanticOutputParser(pydantic_object=Feedback)

str_op = StrOutputParser()

prompt = PromptTemplate(
    template="""
You are a sentiment classifier.

Classify the following feedback as either "positive" or "negative".

Feedback:
{feedback}

{format_instruction}

Return ONLY valid JSON.
""",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = prompt | model | parser

# result = chain.invoke({'feedback': 'This is the best phone ever.'})

# print(result)

positive_prompt = PromptTemplate(
    template='Based on positive review {review} give one best professional thankful message',
    input_variables=['review']
)

negative_prompt = PromptTemplate(
    template='Based on Negative  review {review} write strict message to support team',
    input_variables=['review']
)


branch_chain = RunnableBranch(
    (lambda x: 'positive' in x['sentiment'].lower(), positive_prompt | model | str_op),
    (lambda x: 'negative' in x['sentiment'].lower(), negative_prompt | model | str_op),
    # Add a default fallback branch here:
    lambda x: "Could not determine sentiment classification." 
)


final_chain = chain | branch_chain

result = final_chain.invoke({'feedback': 'This is the best phone ever.'})

print(result)