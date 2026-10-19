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

json_review = {
    "title": "Review",
    "description": "Review of Movie",
    "type": "object",
    "properties": {
        "Movie_name":"object",
        "sentiment":"review_sentiment"
    }
    
}

review = """This film is a masterful blend of absurd comedy, 
stunning sci-fi visuals, and deeply moving family drama. 
It brilliantly uses a chaotic multiverse plot to tell a 
resonant story about generational trauma, depression, and 
the power of radical kindness."""

structured_model = model.with_structured_output(json_review)

result = structured_model.invoke(review)

print(result)