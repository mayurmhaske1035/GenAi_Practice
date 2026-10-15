import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Load .env from the project root
root_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=root_env_path)

# Initialize Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",  # Update from gemini-1.5-flash
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=1024,
)
st.header("Gemini Chat Model with Streamlit")
input_user = st.text_input("Enter your question:")

if st.button("Submit"):
    result = llm.invoke(input_user)
    st.write(result.content)