from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence, RunnableLambda

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template = 'Generate a song in Japanese on  {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "Explain me provided song {song} in Simple English ",
    input_variables = ['song']
)

song_gen_chain = RunnableSequence(prompt1, model, parser)

par_chain = RunnableParallel({
    'song' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(song_gen_chain, par_chain)

result = final_chain.invoke('School life')

print(result)

print('$'*80)

print(result['song'])
print("-"*80)
print(result['explanation'])