import os
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load .env from root directory
root_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=root_env_path)

# Initialize Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",  # Update from gemini-1.5-flash
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=1024,
)

class Review(TypedDict):
    lenght   : int
    sentiment : str
    summary    : str

structured_output = llm.with_structured_output(Review)

result = structured_output.invoke("""Twenty-seven years after 
their first encounter with the terrifying Pennywise, 
the Losers Club have grown up and moved away, 
until a devastating phone call brings them back.""")
print(result)