from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal
from langchain_core.output_parsers import PydanticOutputParser



load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )

class Review(BaseModel):
    key_themes: list[str] = Field(description='Key themes discussed in the review')
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal['pos', 'neg'] = Field(description="Sentiment: pos or neg")
    pros: Optional[list[str]] = Field(default=None)
    cons: Optional[list[str]] = Field(default=None)
    name: Optional[str] = Field(default=None)

parser = PydanticOutputParser(pydantic_object=Review)

prompt_template = PromptTemplate(
    template = "Analyze the following Review.\n {format_instructions} \n Review : {review}",
    input_variables=['review'],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

# print(prompt_template.invoke({'review':'This tablet is very amazing.'}))

review = """I m a long-time patient of this hospital; 
every time I come in, I have to wait over an hour to get seen. 
The staff is nice, and the new renovations look good, 
but I didn't leave work early to wait an hour to be seen
Review by Mayur Mhaske """

prompt = prompt_template.invoke({'review':review})

result = model.invoke(prompt)

print(result.content)