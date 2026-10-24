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

def word_count(text):
    return len(text.split())

#word_runnable = RunnableLambda(word_count)

#print(word_runnable.invoke('India won world cup 2011'))

prompt1 = PromptTemplate(
    template = 'Generate a News on  {topic}',
    input_variables = ['topic']
)

news_gen_chain = RunnableSequence(prompt1, model, parser)

per_chain = RunnableParallel({
    'news': RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(news_gen_chain, per_chain)

result = final_chain.invoke('India won world cup 2011')
print(result)

print('-'*80)
print(result['news'])

print("-"* 80)
print(result['word_count'])