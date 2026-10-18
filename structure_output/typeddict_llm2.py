import os
from typing import TypedDict,Annotated ,Optional 
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

class review(TypedDict):
    key_theme : Annotated[list[str], 'mention key points in review in list ']
    summary : Annotated[str,'Give a brief summary of the review in 1-2 lines']
    sentiment: Annotated[str,'Mention review sentinment in one word and latters as well like positive 1, negative 0']
    prons : Annotated[Optional[list[str]], 'List of pros mentioned in the review if any']
    cons : Annotated[Optional[list[str]], 'List of cons mentioned in the review if any']

structured_output = llm.with_structured_output(review)

result = structured_output.invoke(''' I m a long-time patient of this hospital; 
 every time I come in, I have to wait over an hour to get seen. 
 The staff is nice, and the new renovations look good, 
 but I didn't leave work early to wait an hour to be seen
 Review by Mayur Mhaske ''' )
print(result)