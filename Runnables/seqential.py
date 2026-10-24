from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )

prompt1 = PromptTemplate(
    template = 'Write a song on a {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'Explain me the following content {song}',
    input_variables = ['song']
)

seq_chain = RunnableSequence(prompt1, model,parser,prompt2, model, parser)

result = seq_chain.invoke({'topic': 'Persian Cat'})

print(result)