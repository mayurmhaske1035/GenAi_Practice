import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env from the project root
root_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=root_env_path)

# Initialize console
console = Console()

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0.3,
    max_output_tokens=1024,
)
template = PromptTemplate(template= """
You are good {role} Assistant ,
Answer the following question using this format:
- Give a clear heading.
- Explain the solution briefly.
- Provide correct Python code in a code block.
- Explain the code step by step.
- Show the expected output.,
question:{question}

""",
input_variables= ["role","question"]

)
prompt = template.invoke({"role": "doctor", "question": "give me tips for a healthy lifestyle"})
result = llm.invoke(prompt)
print(result)
print("-*-" * 30)
print(result.content)