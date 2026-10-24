from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence, RunnableBranch

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate a News on  {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Summarize above topic in 2 lines {topic}",
    input_variables = ['topic']
)

news_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x :  len(x.split()) > 50 , RunnableSequence(prompt2, model, parser)),  # if the topic has more than 50 words, summarize it
    RunnablePassthrough()  # else, just return the original topic
)

final_chain = RunnableSequence(news_chain, branch_chain)

result = final_chain.invoke('India won world cup 2011')

print(result)