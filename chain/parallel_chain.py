from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

import streamlit  as st

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


model1 = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )

model2 = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GEMINI_API_KEY,
    )

prompt1= PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2= PromptTemplate(
    template='Generate 5 short questions answers from the following text \n {text}',
    input_variables=['text']
)

prompt3= PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes --> {notes} \n  quiz  {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser ,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

text = """ The life cycle of a Forward Deployed Engineer (FDE) follows an end-to-end customer engagement model—moving from initial problem discovery to production deployment and eventual productization feedback. Unlike standard software engineers who build inside their own company's core product team, an FDE embeds directly with the customer to navigate messy enterprise environments.

The FDE Engagement Lifecycle

A typical FDE project spans roughly 14 to 16 weeks (or a structured 3-to-6-month deployment loop) broken into distinct phases:

1. Discovery & Technical Scoping (Weeks 1–3)

• Unpack real workflows: Audit the customer’s actual operational pain points instead of accepting vague feature requests.
• Map data and systems: Inspect legacy databases, internal APIs, security constraints, and compliance requirements.
• Define explicit exit criteria: Agree on measurable targets with the customer (e.g., proof-of-concept achieving ≥ 85% accuracy on a test set) before writing code.

2. Prototyping & Integration (Weeks 4–8)

• Build rapid proof-of-concepts (POCs): Connect core software or AI components (such as RAG pipelines, agentic workflows, or custom connectors) to sample customer data.
• Handle edge cases: Solve real-world data fragmentation, permission boundaries, and legacy system quirks.
• Validate performance: Run evaluation datasets, latency benchmarks, and human-in-the-loop (HITL) checks.

3. Production Hardening & Deployment (Weeks 9–14)

• Harden for enterprise scale: Implement authentication, audit logging, monitoring, alerts, and security/compliance guardrails.
• Deploy into customer environments: Move the solution from sandbox/test environments directly into the client's cloud or on-premise infrastructure.
• Enable user adoption: Train the customer's internal team, deliver runbooks, and iron out friction during live rollout.

4. Upstream Productization & Hand-Off (Weeks 15–16)

• Iterate and optimize: Track live system usage, debug production failures, and refine performance based on daily user feedback.
• Feed insights back to core product: Translate custom one-off fixes or recurring customer requests into generalized features for the main engineering roadmap.
• Handover or transition: Transition day-to-day operations to the customer's internal engineering team or rotate to the next high-impact deployment.

Core Loop & Career Progression

• The Loop-Back: If a deployment hits blocks between the POC and production stage—where many enterprise AI or software projects stall due to trust or integration gaps—the FDE loops back to redesign evaluation pipelines or fine-tune integrations.
• Career Path: Practitioners typically evolve from an entry/mid FDE to Senior FDE, Staff/Lead FDE, Principal FDE, or transition into engineering leadership, solutions architecture, or head of deployment roles. You can explore the full breakdown on the GSDC FDE Career Path Guide.
If you'd like, let me know:
• Are you preparing for an FDE interview loop (such as Palantir or OpenAI)?
• Or are you trying to structure an FDE team inside your own organization? """

result = final_chain.invoke({'text': text})

print(result)

final_chain.get_graph().print_ascii()