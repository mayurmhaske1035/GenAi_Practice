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
    template = 'Generate a paper based News on  {topic} for Gemini NEWS PUNE',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a News for digital blog  {topic} for GEMINI NEWS PUNE',
    input_variables = ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'paper_news' : RunnableSequence(prompt1, model, parser),
    'digital_news' : RunnableSequence(prompt2, model, parser)
}
)

result = parallel_chain.invoke('India won world cup 2011')

print(result)

print('-'*80)

print(result['paper_news']) 
print("-"*80)
print(result['digital_news']) 