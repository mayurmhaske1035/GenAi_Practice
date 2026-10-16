import os
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

template = PromptTemplate(
    
    template= "give me Result in format which would be easy to read and use heading and bullets  :{question}",
    input_variables=["question"],
    
)
#prompt = template.format_prompt(question="Give me basic rules for addition ")
prompt= template.invoke({ "question": "Give me basic health tips for a healthy lifestyle"})
print(prompt)
result = llm.invoke(prompt)
print(result.content)