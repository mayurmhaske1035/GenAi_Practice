import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Load environment variables from .env file
load_dotenv()

# 2. Configure the Hugging Face Endpoint
# We point directly to the Llama-3.1-8B-Instruct repository
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

# 3. Wrap it in ChatHuggingFace to support system/user message templates natively
chat_model = ChatHuggingFace(llm=llm)

# 4. Formulate your message payload
messages = [
    SystemMessage(content="You are a helpful financial assistant."),
    HumanMessage(content="Tell me what is Nasdaq and how it works?")
]

# 5. Invoke the model and print the response
print("Sending request to Hugging Face...")
response = chat_model.invoke(messages)

print("\nAI Response:")
print(response.content)