# RAG Agent Repository and PDF Generation Instructions

This file contains the complete repository structure for the RAG chatbot from Lesson 2, along with a LaTeX file to generate the PDF of the career transition roadmap and lessons, and a script to automate zipping and PDF creation. Follow the instructions at the end to set up the repository and generate the deliverables.

## Repository Structure

```
rag-agent/
├── data/                          # Folder for sample docs (PDF, TXT, CSV)
├── ingest.py                      # Ingest docs, chunk, embed, upsert to Pinecone
├── agent.py                       # LangChain retriever + conversational agent
├── app_streamlit.py               # Streamlit chat UI
├── app_fastapi.py                 # FastAPI chat API
├── requirements.txt                # Python dependencies
├── Dockerfile                     # Docker configuration for Cloud Run
├── deploy.sh                      # Script to deploy to Cloud Run
├── ai-agent-bootcamp.tex          # LaTeX file for PDF generation
├── generate_zip_pdf.sh            # Script to create zip and PDF
└── README.md                      # Project overview
```

## File Contents

### `ingest.py`
```python
# ingest.py
import os
import glob
import json
from typing import List
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import pdfplumber
import csv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. "us-west1-gcp"
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")

if not OPENAI_KEY or not PINECONE_API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY and PINECONE_API_KEY in env")

# init clients
client = OpenAI(api_key=OPENAI_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# create index if not exists (dense vectors using dimension from OpenAI model e.g. 1536)
if PINECONE_INDEX not in pinecone.list_indexes():
    pinecone.create_index(name=PINECONE_INDEX, dimension=1536)  # adjust dim per model
index = pinecone.Index(PINECONE_INDEX)

def extract_text_from_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def extract_text_from_csv(path: str) -> str:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(" | ".join(r))
    return "\n".join(rows)

def load_documents(data_dir: str) -> List[dict]:
    docs = []
    for p in glob.glob(os.path.join(data_dir, "**/*"), recursive=True):
        if os.path.isdir(p):
            continue
        ext = Path(p).suffix.lower()
        if ext in [".pdf"]:
            text = extract_text_from_pdf(p)
        elif ext in [".txt", ".md"]:
            text = Path(p).read_text(encoding="utf-8")
        elif ext in [".csv"]:
            text = extract_text_from_csv(p)
        else:
            print(f"Skipping unsupported file {p}")
            continue
        docs.append({"source": p, "text": text})
    return docs

def chunk_and_embed_and_upsert(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_vectors = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            # get embedding from OpenAI via langchain or openai client
            emb_resp = client.embeddings.create(model="text-embedding-3-small", input=chunk)
            vector = emb_resp.data[0].embedding
            metadata = {"source": doc["source"], "chunk": i}
            # unique id
            vid = f"{Path(doc['source']).stem}-{i}"
            all_vectors.append((vid, vector, metadata))
    # upsert into pinecone in batches
    batch_size = 100
    for i in range(0, len(all_vectors), batch_size):
        batch = all_vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    print("Upserted", len(all_vectors), "vectors.")

if __name__ == "__main__":
    docs = load_documents("data")
    print(f"Found {len(docs)} docs")
    chunk_and_embed_and_upsert(docs)
```

### `agent.py`
```python
# agent.py
import os
from openai import OpenAI
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# create vectorstore wrapper
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-3-small")
index = pinecone.Index(PINECONE_INDEX)
vectorstore = Pinecone(index, embeddings.embed_query, "text")  # "text" is optional

# chat model
chat = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_KEY)

# conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(chat, vectorstore.as_retriever(search_kwargs={"k":4}), memory=memory)

def answer_query(question: str):
    res = qa_chain({"question": question})
    # res contains 'answer' and chat_history
    return res
```

### `app_streamlit.py`
```python
# app_streamlit.py
import streamlit as st
from agent import answer_query

st.set_page_config(page_title="RAG Chat Demo")
st.title("RAG Chat Demo")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about the documents:")

if st.button("Send") and query:
    res = answer_query(query)
    answer = res.get("answer") or "No answer"
    st.session_state.history.append(("user", query))
    st.session_state.history.append(("bot", answer))
    st.write("**Answer:**")
    st.write(answer)

for role, text in st.session_state.history[::-1]:
    if role == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Agent:** {text}")
```

### `app_fastapi.py`
```python
# app_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent import answer_query
import uvicorn

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def query(q: Query):
    res = answer_query(q.question)
    return {"answer": res.get("answer"), "source_docs": res.get("source_documents", [])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### `requirements.txt`
```
langchain>=0.0.400
openai>=1.0.0
pinecone-client>=5.0.0
tiktoken
python-dotenv
streamlit
fastapi
uvicorn[standard]
requests
pdfplumber
pypdf
transformers
```

### `Dockerfile`
```dockerfile
# Use slim python
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# system deps for pdfplumber
RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config python3-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

# for fastapi
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
```

### `deploy.sh`
```bash
#!/usr/bin/env bash
set -e

PROJECT_ID="${GCP_PROJECT}"
SERVICE_NAME="rag-agent"
REGION="us-central1"

gcloud config set project $PROJECT_ID
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME
gcloud run deploy $SERVICE_NAME --image gcr.io/$PROJECT_ID/$SERVICE_NAME --platform managed --region $REGION --allow-unauthenticated --memory=1Gi
```

### `ai-agent-bootcamp.tex`
```latex
% ai-agent-bootcamp.tex
\documentclass[a4paper,12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{parskip}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{titlesec}

% Define code listing style
\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    numbers=left,
    numberstyle=\tiny,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{red}
}

% Customize headings
\titleformat{\section}{\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{1em}{}

% Use Noto Serif for consistent font rendering
\usepackage{noto}

\begin{document}

\title{AI Agent Career Transition Bootcamp}
\author{Grok}
\date{August 11, 2025}
\maketitle

\tableofcontents
\newpage

\section{Career Transition Roadmap}
Got it — you want to go from being a cloud-focused data pipeline engineer to someone who can \textbf{build, deploy, and monetize AI agent systems} end-to-end, capable of handling your own clients. We’ll treat this like a \textbf{career acceleration bootcamp} with a clear roadmap.

Here’s the \textbf{structured curriculum} I’d design for you, considering your GCP/AWS background and goal to be self-sufficient:

\section{Phase 0 – Foundation Setup (1 week)}
\textbf{Goals:} Understand what AI agents are, their components, and the current market landscape.

\begin{itemize}
    \item \textbf{Concepts to Master}
    \begin{itemize}
        \item What AI agents are (autonomous workflows powered by LLMs + tools)
        \item Core components: LLM, vector DB, memory, tool use, orchestration, connectors
        \item Where they’re used: customer support, research, automation, data analysis, integrations
        \item The AI agent ecosystem: LangChain, CrewAI, Autogen, OpenAI Functions, etc.
        \item Business use cases \& revenue models
    \end{itemize}
    \item \textbf{Action Items}
    \begin{itemize}
        \item Read “AI Agents 101” articles from LangChain docs
        \item Watch 2–3 intro videos from \textit{LangChain, CrewAI, and Microsoft Autogen}
        \item Set up your \textbf{local Python + VSCode environment}
        \item Open accounts: OpenAI API, Anthropic, LangSmith (for debugging), Pinecone/Weaviate
    \end{itemize}
\end{itemize}

\section{Phase 1 – Core Technical Skills (3–4 weeks)}
\textbf{Goal:} Be able to build and run simple AI agents that can reason, call APIs, and maintain context.

\subsection{Python for AI Agents}
\begin{itemize}
    \item Refresher: async programming, API requests, decorators, OOP basics
    \item Libraries: \texttt{openai}, \texttt{langchain}, \texttt{crewai}, \texttt{pydantic}, \texttt{fastapi}
\end{itemize}

\subsection{LLM APIs}
\begin{itemize}
    \item OpenAI GPT-4o, Anthropic Claude 3, Gemini on GCP
    \item Prompt engineering basics $\rightarrow$ advanced (few-shot, chain-of-thought, self-reflection prompts)
    \item Function calling / tool use
\end{itemize}

\subsection{LangChain or CrewAI Basics}
\begin{itemize}
    \item Build your first agent that:
    \begin{itemize}
        \item Answers from docs using retrieval
        \item Calls an API to fetch live data
        \item Summarizes results into a report
    \end{itemize}
\end{itemize}

\subsection{Memory \& Knowledge}
\begin{itemize}
    \item Vector DB basics (Pinecone, Weaviate, Chroma)
    \item Embedding models and semantic search
    \item Document ingestion \& retrieval-augmented generation (RAG)
\end{itemize}

\textbf{Mini-Project:} Build a \textit{Data Transformation QA Agent} that can read CSVs, transform them, and answer business queries.

\section{Phase 2 – Advanced AI Agent Systems (4–6 weeks)}
\textbf{Goal:} Build multi-agent systems with real-world integrations.

\subsection{Multi-Agent Orchestration}
\begin{itemize}
    \item CrewAI / Microsoft Autogen for role-based agents
    \item Agent communication \& coordination patterns
\end{itemize}

\subsection{Tools \& APIs Integration}
\begin{itemize}
    \item Google Cloud APIs (BigQuery, Storage, Vertex AI)
    \item AWS APIs (S3, Lambda)
    \item External SaaS APIs (HubSpot, Slack, Notion, Trello)
\end{itemize}

\subsection{Custom Tools}
\begin{itemize}
    \item Build custom functions to plug into agents
    \item Handling failures, retries, fallbacks
\end{itemize}

\subsection{Data Workflows with AI}
\begin{itemize}
    \item AI-driven ETL transformations
    \item AI + SQL query generation \& validation
\end{itemize}

\textbf{Mini-Project:} Multi-agent system where:
\begin{itemize}
    \item One agent fetches data from BigQuery
    \item Another cleans/transforms it
    \item Another generates business insights \& sends a Slack report
\end{itemize}

\section{Phase 3 – Deployment \& Scalability (3–4 weeks)}
\textbf{Goal:} Run production-ready AI agents on cloud infra you already know.

\begin{itemize}
    \item \textbf{Containerizing AI Agents}
    \begin{itemize}
        \item Dockerize LangChain/CrewAI projects
        \item Use Cloud Run (GCP) or Lambda/Fargate (AWS)
    \end{itemize}
    \item \textbf{Serverless APIs}
    \begin{itemize}
        \item Deploy AI agent APIs with FastAPI on Cloud Run
        \item Rate limiting, logging, error handling
    \end{itemize}
    \item \textbf{Monitoring \& Debugging}
    \begin{itemize}
        \item LangSmith tracing
        \item Vector DB performance tuning
        \item LLM cost optimization
    \end{itemize}
\end{itemize}

\textbf{Mini-Project:} Deploy a RAG chatbot on GCP Cloud Run, integrated with a client’s knowledge base.

\section{Phase 4 – Productization \& Business Skills (2–3 weeks)}
\textbf{Goal:} Learn how to get clients, sell solutions, and run as a one-person AI consultancy.

\subsection{Identify Niche Markets}
\begin{itemize}
    \item Your cloud/data background $\rightarrow$ AI for \textbf{data reporting automation}, \textbf{financial insights}, \textbf{compliance checks}
    \item Micro-SaaS opportunities
\end{itemize}

\subsection{Client Project Flow}
\begin{itemize}
    \item Discovery $\rightarrow$ Proposal $\rightarrow$ Build $\rightarrow$ Deploy $\rightarrow$ Maintain
    \item Writing clear SOW (Statement of Work)
    \item Pricing models (fixed bid, hourly, subscription)
\end{itemize}

\subsection{Selling Platforms}
\begin{itemize}
    \item LinkedIn thought leadership
    \item Upwork / Fiverr premium gigs
    \item Direct outreach to startups
    \item Partnering with agencies
\end{itemize}

\subsection{Legal \& Operations}
\begin{itemize}
    \item Contracts \& NDAs
    \item Data privacy compliance
    \item Billing \& invoicing
\end{itemize}

\textbf{Mini-Project:} Build a \textit{demo AI agent product} + publish a case study on LinkedIn.

\section{Phase 5 – Mastery \& Scaling (Ongoing)}
\begin{itemize}
    \item Build portfolio of 3–5 public demos
    \item Explore advanced areas:
    \begin{itemize}
        \item Fine-tuning LLMs
        \item Hybrid search (symbolic + neural)
        \item Edge deployment for low-latency agents
    \end{itemize}
    \item Offer AI automation retainers to clients
\end{itemize}

\section{Tech Stack Recommendation}
\begin{itemize}
    \item \textbf{Agent Frameworks:} LangChain + CrewAI + Autogen
    \item \textbf{LLMs:} OpenAI GPT-4o, Claude 3.5, Gemini 1.5
    \item \textbf{Vector DBs:} Pinecone, Weaviate
    \item \textbf{Hosting:} GCP Cloud Run, AWS Lambda
    \item \textbf{Frontend:} Streamlit or Next.js for quick UIs
    \item \textbf{Monitoring:} LangSmith, Prometheus, Cloud Logging
\end{itemize}

\section{Next Steps}
If you want, I can \textbf{turn this into a 12-week structured learning plan with weekly projects and milestones} so you have an exact execution path. That way, by the end, you’ll not only know how AI agents work, but also have \textbf{marketable demos} and a \textbf{client acquisition system} in place.

Do you want me to break this into that \textbf{week-by-week bootcamp plan} next?

\section{Introduction to Lessons}
Fantastic — love the energy, Vishul. I’ll teach you \textbf{start $\rightarrow$ finish}, hands-on, step-by-step, with small projects, runnable commands, code examples, and deliverables you can show clients. I’ll also call out what to deploy on GCP (since that’s your main stack) and alternates for AWS when useful.

Below is a \textbf{concrete learning workflow} and then I’ll \textit{teach Lesson 0 and Lesson 1 right now} (the immediate, practical stuff). After you go through these, tell me whether you want me to continue lesson-by-lesson in this chat (I’ll keep giving code, infra steps and exercises), or if you’d like a downloadable schedule (I can produce a week-by-week checklist and sample SOW).

\section{Big Picture (Short)}
AI agents = LLM \textbf{reasoning + tool use + memory + orchestration + infra + observability}. Popular frameworks and services to know: \textbf{LangChain} (agents), \textbf{OpenAI function-calling}, \textbf{vector DBs like Pinecone}, observability (LangSmith), and multi-agent frameworks such as \textbf{Microsoft AutoGen}, \textbf{CrewAI}, and cloud offerings like \textbf{Vertex AI Agent Engine}. (\href{https://python.langchain.com/api_reference/core/agents.html}{LangChain}, \href{https://platform.openai.com/docs/guides/function-calling}{OpenAI Platform}, \href{https://docs.pinecone.io/}{Pinecone Docs}, \href{https://docs.smith.langchain.com/observability}{LangSmith}, \href{https://microsoft.github.io/autogen/stable//index.html}{Microsoft GitHub}, \href{https://github.com/crewAIInc/crewAI}{GitHub}, \href{https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview}{Google Cloud})

\section{Learning Workflow}
\begin{enumerate}
    \item I’ll give short lessons with: concept, hands-on steps (commands \& code), small project, tests/checklist, recommended reading.
    \item You do the hands-on, paste logs/errors here if anything breaks, I debug \& iterate.
    \item After every 2 lessons you’ll build a mini product (demo) you can show a client.
\end{enumerate}

\section{Bootcamp Roadmap (12 Weeks — Brief)}
\begin{itemize}
    \item \textbf{Week 0:} Setup \& foundations (accounts, env, basic concepts) — \textit{we start this now}.
    \item \textbf{Weeks 1–3:} Core agent building (LLM API, function calling, LangChain/CrewAI basics, RAG).
    \item \textbf{Weeks 4–6:} Multi-agent orchestration, tool building, memory and vector DBs, secure connectors.
    \item \textbf{Weeks 7–8:} Production infra — containerization, Cloud Run / Vertex / Cloud infra, monitoring, cost control.
    \item \textbf{Weeks 9–10:} Productize — UI, SSO, pricing model, SOW \& contracts.
    \item \textbf{Weeks 11–12:} Sales \& scale — demos, outreach playbook, case studies, recurring revenue setups.
\end{itemize}

If that plan looks good I’ll keep the tempo. Now — \textbf{Lesson 0 + Lesson 1}.

\section{Lesson 0 — Setup \& Accounts}
\textbf{Goal:} working local dev environment, API keys, and basic tools installed.

\textbf{What you need to create / configure:}
\begin{itemize}
    \item Accounts: \textbf{OpenAI} (for LLMs \& embeddings), \textbf{Pinecone} (or alternative vector DB), \textbf{LangSmith} (observability — optional but highly recommended), and your \textbf{GCP} project with Cloud Run/Vertex permissions. (You can use AWS equivalents later.) (\href{https://platform.openai.com/docs/guides/function-calling}{OpenAI Platform}, \href{https://docs.pinecone.io/}{Pinecone Docs}, \href{https://docs.smith.langchain.com/}{LangSmith}, \href{https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview}{Google Cloud})
\end{itemize}

\textbf{Terminal commands (macOS / Linux):} Run these in a project folder (replace names as needed):
\begin{lstlisting}[language=bash]
# create project folder
mkdir ai-agents-bootcamp && cd ai-agents-bootcamp

# python virtualenv recommended
python3 -m venv .venv
source .venv/bin/activate

# upgrade pip
pip install --upgrade pip

# install core libs (we'll adjust per lesson)
pip install langchain openai pinecone-client tiktoken python-dotenv fastapi uvicorn
\end{lstlisting}

\textbf{Create a \texttt{.env} in project root and add (fill keys from the UIs you created):}
\begin{lstlisting}
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pc-...
PINECONE_ENV=us-west1-gcp
PINECONE_INDEX=my-index
LANGSMITH_API_KEY=ls-...
GCP_PROJECT=your-gcp-project-id
\end{lstlisting}

\textbf{Quick checklist:}
\begin{itemize}
    \item [ ] Python 3.10+ environment active
    \item [ ] OPENAI + Pinecone accounts + API keys
    \item [ ] GCP project + gcloud CLI installed \& authenticated
    \item [ ] (Optional) LangSmith account for tracing
\end{itemize}

\section{Lesson 1 — Core Concepts \& First Tool-Calling Agent}
\textbf{Goal:} Understand agent components and build a tiny agent that \textbf{asks a model, the model asks to call a function, you run that function and feed result back} (the classic tool / function-calling pattern). This is the basis of safe tool integration and agent orchestration. (We’ll build retrieval and LangChain agents next.)

\subsection{Quick Theory (Short and Precise)}
An AI agent system commonly contains:
\begin{itemize}
    \item \textbf{LLM} — the model that reasons and produces text or structured output. (\href{https://python.langchain.com/api_reference/core/agents.html}{LangChain})
    \item \textbf{Tools / Functions} — executable APIs the agent can call (search, DB query, run SQL, invoke Lambda, etc.). LLMs can be instructed to return a structured “function call” output which your runtime executes. (\href{https://platform.openai.com/docs/guides/function-calling}{OpenAI Platform})
    \item \textbf{Memory / Knowledge} — persistent grounding (embeddings stored in a vector DB) used for retrieval-augmented generation (RAG). Vector DBs (Pinecone, Weaviate, Qdrant, etc.) store embeddings and enable similarity search. (\href{https://docs.pinecone.io/}{Pinecone Docs})
    \item \textbf{Orchestrator / Agent framework} — organizes thought $\rightarrow$ action cycles, manages tool selection and state (LangChain, CrewAI, AutoGen). (\href{https://python.langchain.com/api_reference/core/agents.html}{LangChain}, \href{https://github.com/crewAIInc/crewAI}{GitHub}, \href{https://microsoft.github.io/autogen/stable//index.html}{Microsoft GitHub})
    \item \textbf{Observability} — traces, runs, and test results (LangSmith) so you can debug agent decisions and costs. (\href{https://docs.smith.langchain.com/observability}{LangSmith})
\end{itemize}

\subsection{Hands-on Lab: Function-Calling with a Simple Weather Tool (curl + Python)}
We’ll do a minimal end-to-end example using the LLM’s function-calling mechanism (model returns JSON indicating which function to call) and you executing the function. This pattern is universal (works with OpenAI-style APIs). I’ll show a \texttt{curl} example (works without any SDK), then a small Python snippet to follow up.

\textbf{Step A — curl (make sure \texttt{\$OPENAI_API_KEY} is exported)}
\begin{lstlisting}[language=bash]
export OPENAI_API_KEY="sk-...yourkey..."

curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o", 
    "messages": [{"role":"user","content":"What is the weather in Pune tomorrow?"}],
    "functions": [
      {
        "name": "get_weather",
        "description": "Get the weather for a location and date",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type":"string"},
            "date": {"type":"string", "description":"YYYY-MM-DD"}
          },
          "required": ["location","date"]
        }
      }
    ],
    "function_call": "auto"
  }'
\end{lstlisting}

\textbf{What happens:}
\begin{enumerate}
    \item Model returns either text or a \texttt{function_call} JSON indicating \texttt{name} and \texttt{arguments}.
    \item Your code should detect \texttt{function_call}, parse \texttt{arguments}, run the real \texttt{get_weather(...)} function (you implement it — real API call or mock), then pass the function result back to the model in a follow-up message so the model can produce a final answer. This is the standard tool pattern. (\href{https://platform.openai.com/docs/guides/function-calling}{OpenAI Platform})
\end{enumerate}

\textbf{Step B — minimal Python skeleton (pseudo-runnable)}
\begin{lstlisting}[language=python]
# file: function_call_agent.py
import os, json, requests

OPENAI_KEY = os.environ["OPENAI_API_KEY"]
HEADERS = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}

def call_openai(messages, functions=None):
    body = {"model":"gpt-4o", "messages": messages}
    if functions:
        body["functions"] = functions
        body["function_call"] = "auto"
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=body)
    return resp.json()

def get_weather(location, date):
    # For the lab, mock or return a fixed JSON. Replace this with real weather API calls.
    return {"location": location, "date": date, "summary": "Sunny, 29C"}

# 1) Ask model
user_msg = {"role":"user","content":"Tell me the weather in Pune on 2025-08-12."}
resp = call_openai([user_msg], functions=[{
    "name":"get_weather",
    "description":"Get the weather for a location and date",
    "parameters":{
        "type":"object",
        "properties":{
            "location":{"type":"string"},
            "date":{"type":"string"}
        },
        "required":["location","date"]
    }
}])

# 2) If model wants to call function, execute it
choice = resp["choices"][0]
if "message" in choice and choice["message"].get("function_call"):
    fc = choice["message"]["function_call"]
    args = json.loads(fc["arguments"])
    result = get_weather(args["location"], args["date"])
    # 3) Send result back to model for final response
    followup = [
        {"role":"user","content":"Please provide the final answer using the function result."},
        {"role":"function","name":fc["name"], "content": json.dumps(result)}
    ]
    final = call_openai(followup)
    print(final["choices"][0]["message"]["content"])
else:
    print(choice["message"]["content"])
\end{lstlisting}

\textbf{Exercise 1 (30–60 min)}
\begin{itemize}
    \item Run the curl example; capture the JSON — find the function_call block.
    \item Implement \texttt{function_call_agent.py} (or a notebook), run it, mock the \texttt{get_weather} function.
    \item When working, paste your terminal output here if you get an error — I’ll debug it with you.
\end{itemize}

\section{Short Deliverable After Lesson 1}
\begin{itemize}
    \item A working script that: sends a user message $\rightarrow$ model requests a function call $\rightarrow$ you execute function $\rightarrow$ feed result back $\rightarrow$ model prints final answer.
    \item Save that script as \texttt{function_call_agent.py} in your repo and commit.
\end{itemize}

\section{Sources for Lesson 0 and 1}
\begin{itemize}
    \item LangChain agents documentation (agents components and patterns). (\href{https://python.langchain.com/docs/tutorials/agents/}{LangChain})
    \item OpenAI function-calling guide (how models can return a function call and how to handle it). (\href{https://platform.openai.com/docs/guides/function-calling}{OpenAI Platform})
    \item Pinecone docs — vector DBs and embeddings for RAG. (\href{https://docs.pinecone.io/}{Pinecone Docs})
    \item LangSmith observability docs — tracing and debugging LLM apps. (\href{https://docs.smith.langchain.com/observability}{LangSmith})
    \item Microsoft AutoGen, CrewAI, Vertex AI Agent Engine — multi-agent and cloud agent infra (for advanced orchestration \& production on GCP). (\href{https://microsoft.github.io/autogen/stable//index.html}{Microsoft GitHub}, \href{https://github.com/crewAIInc/crewAI}{GitHub}, \href{https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview}{Google Cloud})
\end{itemize}

\section{Lesson 2 — Retrieval-Augmented Generation (RAG) Agent}
Awesome — let’s dive into \textbf{Lesson 2: Retrieval-Augmented Generation (RAG)} end-to-end. I’ll teach you everything: document ingestion, embeddings, vector DB (Pinecone), LangChain retriever + agent, a simple chat UI (Streamlit or FastAPI), containerization, and deploy to \textbf{GCP Cloud Run}. I’ll give ready-to-run code, infra commands, and notes for production hardening.

I’ll assume you completed Lesson 0/1 (Python env, \texttt{OPENAI_API_KEY}, \texttt{PINECONE_API_KEY}, \texttt{PINECONE_ENV}, etc.). If not, stop and do that first — I referenced setup in Lesson 0.

Key references I used while assembling this lesson: LangChain RAG tutorial, Pinecone quickstart + RAG guides, OpenAI embeddings docs, and GCP Cloud Run Python quickstart. (\href{https://python.langchain.com/docs/tutorials/rag/}{LangChain}, \href{https://docs.pinecone.io/guides/get-started/build-a-rag-chatbot}{Pinecone Docs}, \href{https://platform.openai.com/docs/guides/embeddings}{OpenAI Platform}, \href{https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service}{Google Cloud})

\subsection{Lesson 2 — Goal}
A working RAG chatbot that:
\begin{enumerate}
    \item Ingests a folder of docs (PDF, TXT, CSV), chunks and creates embeddings (OpenAI embeddings).
    \item Stores embeddings in Pinecone.
    \item Uses LangChain to retrieve relevant passages and an LLM to answer conversational queries (with conversation history).
    \item Serves a small chat UI (Streamlit) or an API (FastAPI).
    \item Dockerized and deployable to Cloud Run.
\end{enumerate}

\subsection{Before We Start — Checklist}
\begin{itemize}
    \item [ ] Python 3.10+ venv active
    \item [ ] \texttt{OPENAI_API_KEY} in env
    \item [ ] \texttt{PINECONE_API_KEY} and \texttt{PINECONE_ENV} in env
    \item [ ] Pinecone index created (we’ll include code to create it programmatically)
    \item [ ] \texttt{PROJECT_ID} set for GCP and \texttt{gcloud} installed \& authenticated for deploy step
\end{itemize}

Helpful docs: Pinecone quickstart and LangChain RAG tutorial. (\href{https://docs.pinecone.io/guides/get-started/quickstart}{Pinecone Docs}, \href{https://python.langchain.com/docs/tutorials/rag/}{LangChain})

\subsection{Project Layout}
Create a folder \texttt{rag-agent/} with files:
\begin{lstlisting}
rag-agent/
├─ ingest.py           # ingest docs -> embeddings -> Pinecone
├─ app_streamlit.py    # Streamlit chat UI (quick demo)
├─ app_fastapi.py      # FastAPI chat API (option)
├─ agent.py            # LangChain agent/retriever logic
├─ requirements.txt
├─ Dockerfile
├─ deploy.sh
└─ README.md
\end{lstlisting}

\subsection{Install Dependencies}
\textbf{\texttt{requirements.txt}} (start simple — add more later)
\begin{lstlisting}
langchain>=0.0.400
openai>=1.0.0
pinecone-client>=5.0.0
tiktoken
python-dotenv
streamlit
fastapi
uvicorn[standard]
requests
pdfplumber
pypdf
transformers  # optional for local tokenizers or extra tools
\end{lstlisting}

\textbf{Install:}
\begin{lstlisting}[language=bash]
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\end{lstlisting}

\subsection{\texttt{ingest.py} — Ingest Docs, Chunk, Embed, Upsert to Pinecone}
This script:
\begin{itemize}
    \item walks \texttt{./data/} folder for files,
    \item extracts text (basic PDF/TXT/CSV handling),
    \item chunks text (simple token/window approach),
    \item generates embeddings with OpenAI,
    \item upserts to Pinecone index with metadata (source, chunk_id).
\end{itemize}

\textbf{ingest.py}
\begin{lstlisting}[language=python]
# ingest.py
import os
import glob
import json
from typing import List
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import pdfplumber
import csv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. "us-west1-gcp"
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")

if not OPENAI_KEY or not PINECONE_API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY and PINECONE_API_KEY in env")

# init clients
client = OpenAI(api_key=OPENAI_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# create index if not exists (dense vectors using dimension from OpenAI model e.g. 1536)
if PINECONE_INDEX not in pinecone.list_indexes():
    pinecone.create_index(name=PINECONE_INDEX, dimension=1536)  # adjust dim per model
index = pinecone.Index(PINECONE_INDEX)

def extract_text_from_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def extract_text_from_csv(path: str) -> str:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(" | ".join(r))
    return "\n".join(rows)

def load_documents(data_dir: str) -> List[dict]:
    docs = []
    for p in glob.glob(os.path.join(data_dir, "**/*"), recursive=True):
        if os.path.isdir(p):
            continue
        ext = Path(p).suffix.lower()
        if ext in [".pdf"]:
            text = extract_text_from_pdf(p)
        elif ext in [".txt", ".md"]:
            text = Path(p).read_text(encoding="utf-8")
        elif ext in [".csv"]:
            text = extract_text_from_csv(p)
        else:
            print(f"Skipping unsupported file {p}")
            continue
        docs.append({"source": p, "text": text})
    return docs

def chunk_and_embed_and_upsert(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_vectors = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            # get embedding from OpenAI via langchain or openai client
            emb_resp = client.embeddings.create(model="text-embedding-3-small", input=chunk)
            vector = emb_resp.data[0].embedding
            metadata = {"source": doc["source"], "chunk": i}
            # unique id
            vid = f"{Path(doc['source']).stem}-{i}"
            all_vectors.append((vid, vector, metadata))
    # upsert into pinecone in batches
    batch_size = 100
    for i in range(0, len(all_vectors), batch_size):
        batch = all_vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    print("Upserted", len(all_vectors), "vectors.")

if __name__ == "__main__":
    docs = load_documents("data")
    print(f"Found {len(docs)} docs")
    chunk_and_embed_and_upsert(docs)
\end{lstlisting}

\textbf{Notes:}
\begin{itemize}
    \item Uses \texttt{text-embedding-3-small} (OpenAI) — pick appropriate model \& dims. See OpenAI embeddings docs. (\href{https://platform.openai.com/docs/guides/embeddings}{OpenAI Platform})
    \item Pinecone index dimension must match embedding size; you can programmatically create integrated indexes via Pinecone APIs. (\href{https://docs.pinecone.io/reference/api/2025-01/control-plane/create_for_model}{Pinecone Docs})
\end{itemize}

\subsection{\texttt{agent.py} — Retriever + Conversational Agent (LangChain)}
This file exposes a function \texttt{answer_query(query, chat_history)} that:
\begin{itemize}
    \item embeds the query,
    \item searches Pinecone,
    \item constructs prompt with top-k contexts,
    \item calls the LLM, preserving conversation history.
\end{itemize}

\textbf{agent.py}
\begin{lstlisting}[language=python]
# agent.py
import os
from openai import OpenAI
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# create vectorstore wrapper
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-3-small")
index = pinecone.Index(PINECONE_INDEX)
vectorstore = Pinecone(index, embeddings.embed_query, "text")  # "text" is optional

# chat model
chat = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_KEY)

# conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(chat, vectorstore.as_retriever(search_kwargs={"k":4}), memory=memory)

def answer_query(question: str):
    res = qa_chain({"question": question})
    # res contains 'answer' and chat_history
    return res
\end{lstlisting}

\textbf{Notes:}
\begin{itemize}
    \item LangChain provides \texttt{ConversationalRetrievalChain} that wires retrieval to an LLM; tutorial here. (\href{https://python.langchain.com/docs/tutorials/rag/}{LangChain})
    \item Adjust \texttt{k} (how many docs) and temperature for deterministic answers.
\end{itemize}

\subsection{Quick Demo UI — \texttt{app_streamlit.py}}
Simple Streamlit chat where user types queries, we call \texttt{agent.answer_query} and display the answer.

\textbf{app_streamlit.py}
\begin{lstlisting}[language=python]
# app_streamlit.py
import streamlit as st
from agent import answer_query

st.set_page_config(page_title="RAG Chat Demo")
st.title("RAG Chat Demo")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about the documents:")

if st.button("Send") and query:
    res = answer_query(query)
    answer = res.get("answer") or "No answer"
    st.session_state.history.append(("user", query))
    st.session_state.history.append(("bot", answer))
    st.write("**Answer:**")
    st.write(answer)

for role, text in st.session_state.history[::-1]:
    if role == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Agent:** {text}")
\end{lstlisting}

\textbf{Run locally:}
\begin{lstlisting}[language=bash]
streamlit run app_streamlit.py
\end{lstlisting}

\subsection{Option: FastAPI API (\texttt{app_fastapi.py})}
If you prefer an API you can call from a frontend or webhook.

\textbf{app_fastapi.py}
\begin{lstlisting}[language=python]
# app_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent import answer_query
import uvicorn

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def query(q: Query):
    res = answer_query(q.question)
    return {"answer": res.get("answer"), "source_docs": res.get("source_documents", [])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
\end{lstlisting}

\textbf{Run locally:}
\begin{lstlisting}[language=bash]
uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8080
\end{lstlisting}

\subsection{Dockerize \& Deploy to Cloud Run}
\textbf{Dockerfile} (simple example for FastAPI + Uvicorn; for Streamlit minor tweaks needed)
\begin{lstlisting}[language=dockerfile]
# Use slim python
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# system deps for pdfplumber
RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config python3-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

# for fastapi
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
\end{lstlisting}

\textbf{deploy.sh}
\begin{lstlisting}[language=bash]
#!/usr/bin/env bash
set -e

PROJECT_ID="${GCP_PROJECT}"
SERVICE_NAME="rag-agent"
REGION="us-central1"

gcloud config set project $PROJECT_ID
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME
gcloud run deploy $SERVICE_NAME --image gcr.io/$PROJECT_ID/$SERVICE_NAME --platform managed --region $REGION --allow-unauthenticated --memory=1Gi
\end{lstlisting}

\textbf{Or use Cloud Run quick deploy:}
\begin{lstlisting}[language=bash]
gcloud run deploy rag-agent --source . --region us-central1 --platform managed --allow-unauthenticated
\end{lstlisting}

\textbf{(GCP can build automatically from source and containerize — quickstart docs here.)} (\href{https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service}{Google Cloud})

\textbf{Notes \& gotchas:}
\begin{itemize}
    \item Keep secrets out of the image. For Cloud Run, use \textbf{Secret Manager} or set env vars in Cloud Run service settings. Never bake API keys into images.
    \item Increase memory for embeddings + processing large docs.
    \item For Streamlit, change CMD and expose port 8080; Streamlit needs \texttt{streamlit run app_streamlit.py --server.port \$PORT --server.address 0.0.0.0}.
\end{itemize}

\subsection{Testing \& Verification}
\begin{enumerate}
    \item Put sample docs in \texttt{data/} (PDFs or text).
    \item Run \texttt{python ingest.py} — check Pinecone console to ensure vectors upserted. Pinecone has quickstart guides. (\href{https://docs.pinecone.io/guides/get-started/quickstart}{Pinecone Docs})
    \item Run \texttt{streamlit run app_streamlit.py} and ask document-specific questions — check that answers include references to sources.
    \item Check logs for inference calls — LangChain and your app should log LLM calls. For production, integrate LangSmith for traces. (\href{https://python.langchain.com/docs/tutorials/rag/}{LangChain}, \href{https://www.youtube.com/watch?v=79e1vm-mTs4}{YouTube})
\end{enumerate}

\subsection{Production Improvements (Next Steps)}
\begin{itemize}
    \item \textbf{Chunking strategy:} Use token-aware splitters and preserve section titles / metadata.
    \item \textbf{Hybrid search:} combine sparse (BM25) + dense embeddings for exact match.
    \item \textbf{Filtering \& namespaces:} add tenant / namespace metadata for multitenancy (Pinecone supports namespaces).
    \item \textbf{Security:} proxy LLM calls via a backend, rate limiting, monitor costs.
    \item \textbf{Observability:} instrument traces and prompt logs with LangSmith or your own logging. (\href{https://www.youtube.com/watch?v=79e1vm-mTs4}{YouTube})
    \item \textbf{Cost control:} cache embeddings, batch embedding requests, curate retrieval top-k.
\end{itemize}

\subsection{Quick FAQ — Common Errors}
\begin{itemize}
    \item \textbf{Dim mismatch when creating Pinecone index}: confirm embedding model dimension and index dimension match. Create index programmatically or via Pinecone console. (\href{https://docs.pinecone.io/guides/indexes/create-an-index}{Pinecone Docs})
    \item \textbf{PDF text extraction empty}: try alternative libraries (PyPDF2, pdfplumber) and test pages for images (OCR needed).
    \item \textbf{Streamlit not reachable on Cloud Run}: ensure you set \texttt{--server.port \$PORT} and \texttt{--server.address 0.0.0.0}. See Cloud Run + Streamlit notes. (\href{https://medium.com/google-cloud/how-to-deploy-your-streamlit-web-app-to-google-cloud-run-with-ease-c9f044aabc12}{Medium}, \href{https://www.youtube.com/watch?v=BGMdxpXsbB4}{YouTube})
\end{itemize}

\subsection{Homework / Mini-Project (Deliverable)}
\begin{enumerate}
    \item Add 5–10 docs relevant to a domain (e.g., finance reports, product docs).
    \item Run \texttt{ingest.py} and confirm vectors in Pinecone.
    \item Run \texttt{streamlit} UI locally and ask 10 domain-specific questions. Save 3 example Q\&A pairs demonstrating RAG behaviour.
    \item Dockerize and deploy the API to Cloud Run with env vars stored via Cloud Run settings (or Secret Manager).
\end{enumerate}

\textbf{When done, paste:}
\begin{itemize}
    \item The console output of \texttt{ingest.py} (or errors),
    \item A sample response from the Streamlit/fastAPI,
    \item The \texttt{gcloud run deploy} output (if you deployed).
\end{itemize}

I’ll review and help harden it for clients (SLA, multi-tenant, costs).

\subsection{Helpful Reading / Tutorials (Quick Links)}
\begin{itemize}
    \item LangChain RAG tutorial. (\href{https://python.langchain.com/docs/tutorials/rag/}{LangChain})
    \item Pinecone RAG \& quickstart. (\href{https://docs.pinecone.io/guides/get-started/build-a-rag-chatbot}{Pinecone Docs})
    \item OpenAI embeddings guide. (\href{https://platform.openai.com/docs/guides/embeddings}{OpenAI Platform})
    \item GCP Cloud Run Python quickstart. (\href{https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service}{Google Cloud})
\end{itemize}

\section{Next Steps for Lesson 2}
Want me to:
\begin{itemize}
    \item[A)] Post a \textbf{fully tested repository} (I’ll produce a zip-style set of files here you can copy/paste) with exact working versions and small improvements (pagination, batching, better chunking)?
    \item[B)] Continue to the next lesson now: \textbf{multi-agent orchestration} (role-based agents, tool chaining, and building an agent that calls BigQuery \& updates datasets)?
\end{itemize}

Say which one and I’ll continue step-by-step.

\section{Sources for Lesson 2}
\begin{itemize}
    \item[1] \href{https://python.langchain.com/docs/tutorials/rag/}{Build a Retrieval Augmented Generation (RAG) App: Part 1}
    \item[2] \href{https://docs.pinecone.io/guides/get-started/build-a-rag-chatbot}{Build a RAG chatbot - Pinecone Docs}
    \item[3] \href{https://platform.openai.com/docs/guides/embeddings}{Vector embeddings - OpenAI API}
    \item[4] \href{https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service}{Build and deploy a Python web app to Google Cloud with Cloud Run}
    \item[5] \href{https://docs.pinecone.io/guides/get-started/quickstart}{Quickstart - Pinecone Docs}
    \item[6] \href{https://docs.pinecone.io/reference/api/2025-01/control-plane/create_for_model}{Create an index with integrated embedding - Pinecone Docs}
    \item[7] \href{https://www.youtube.com/watch?v=79e1vm-mTs4}{FastAPI on Cloud Run - YouTube}
    \item[8] \href{https://docs.pinecone.io/guides/indexes/create-an-index}{Create a serverless index - Pinecone Docs}
    \item[9] \href{https://medium.com/google-cloud/how-to-deploy-your-streamlit-web-app-to-google-cloud-run-with-ease-c9f044aabc12}{How to deploy your Streamlit Web App to Google Cloud Run with ...}
    \item[10] \href{https://www.youtube.com/watch?v=BGMdxpXsbB4}{Deploy a Streamlit App with Google Cloud Run, within ... - YouTube}
\end{itemize}

\end{document}
```

### `generate_zip_pdf.sh`
```bash
#!/usr/bin/env bash
set -e

# Create rag-agent directory and subdirectories
mkdir -p rag-agent/data

# Copy files (assuming they are already created in the current directory)
cp ingest.py agent.py app_streamlit.py app_fastapi.py requirements.txt Dockerfile deploy.sh ai-agent-bootcamp.tex rag-agent/
cp README.md rag-agent/

# Install Pandoc and LaTeX if not already installed
if ! command -v pandoc &> /dev/null; then
    echo "Installing Pandoc..."
    sudo apt-get update && sudo apt-get install -y pandoc
fi
if ! command -v xelatex &> /dev/null; then
    echo "Installing TeX Live..."
    sudo apt-get install -y texlive texlive-fonts-extra
fi

# Generate PDF
cd rag-agent
xelatex ai-agent-bootcamp.tex
xelatex ai-agent-bootcamp.tex  # Run twice to ensure TOC is generated
cd ..

# Create zip
zip -r rag-agent.zip rag-agent/

echo "Generated rag-agent.zip and ai-agent-bootcamp.pdf in rag-agent/"
```

### `README.md`
```markdown
# RAG Chatbot Project

This project implements a Retrieval-Augmented Generation (RAG) chatbot that ingests documents (PDF, TXT, CSV), stores embeddings in Pinecone, and answers queries using LangChain. It includes both a Streamlit UI and a FastAPI endpoint, deployable to GCP Cloud Run.

## Setup
1. Create a `.env` file with:
   ```
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=pc-...
   PINECONE_ENV=us-west1-gcp
   PINECONE_INDEX=rag-index
   GCP_PROJECT=your-gcp-project-id
   ```
2. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Add sample documents to `data/` folder.
4. Run `python ingest.py` to process documents.
5. Run the app:
   - Streamlit: `streamlit run app_streamlit.py`
   - FastAPI: `uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8080`
6. Deploy to Cloud Run:
   ```bash
   bash deploy.sh
   ```

## Files
- `ingest.py`: Ingests documents, chunks, embeds, and upserts to Pinecone.
- `agent.py`: LangChain-based conversational RAG agent.
- `app_streamlit.py`: Streamlit UI for chat.
- `app_fastapi.py`: FastAPI API for programmatic access.
- `Dockerfile`: Docker configuration for Cloud Run.
- `deploy.sh`: Deployment script for GCP Cloud Run.
- `ai-agent-bootcamp.tex`: LaTeX file for PDF documentation.
- `generate_zip_pdf.sh`: Script to create zip and PDF.

## Notes
- Ensure Pinecone index dimension matches the embedding model (e.g., 1536 for `text-embedding-3-small`).
- Use GCP Secret Manager for sensitive keys in production.
- See `ai-agent-bootcamp.pdf` for the full career roadmap and lessons.
```

## Instructions to Create the Repository and PDF

1. **Create the Directory Structure**:
   ```bash
   mkdir rag-agent
   mkdir rag-agent/data
   ```

2. **Copy Files**:
   - Create each file (`ingest.py`, `agent.py`, `app_streamlit.py`, `app_fastapi.py`, `requirements.txt`, `Dockerfile`, `deploy.sh`, `ai-agent-bootcamp.tex`, `generate_zip_pdf.sh`, `README.md`) in the `rag-agent/` directory.
   - Copy the content from each section above into the respective file using a text editor (e.g., VSCode).
   - Ensure file permissions for `deploy.sh` and `generate_zip_pdf.sh`:
     ```bash
     chmod +x rag-agent/deploy.sh rag-agent/generate_zip_pdf.sh
     ```

3. **Install Dependencies for PDF Generation**:
   - Install Pandoc and TeX Live (includes XeLaTeX):
     ```bash
     # On Ubuntu
     sudo apt-get update
     sudo apt-get install -y pandoc texlive texlive-fonts-extra
     # On macOS
     brew install pandoc
     brew install --cask mactex
     ```
   - Verify installations:
     ```bash
     pandoc --version
     xelatex --version
     ```

4. **Run the Generation Script**:
   - Navigate to the parent directory containing `rag-agent/`:
     ```bash
     cd rag-agent
     bash generate_zip_pdf.sh
     ```
   - This will:
     - Generate `ai-agent-bootcamp.pdf` inside `rag-agent/`.
     - Create `rag-agent.zip` containing the entire `rag-agent/` directory, including the PDF.

5. **Verify Outputs**:
   - Check `rag-agent/ai-agent-bootcamp.pdf` to ensure all sections, headings, code blocks, and links are preserved.
   - Unzip `rag-agent.zip` to verify all files are included:
     ```bash
     unzip rag-agent.zip -d test-unzip
     ```

6. **Test the RAG Chatbot**:
   - Create a `.env` file in `rag-agent/` with your API keys (as shown in `README.md`).
   - Add sample documents (e.g., PDFs, TXTs) to `rag-agent/data/`.
   - Run:
     ```bash
     cd rag-agent
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     python ingest.py
     streamlit run app_streamlit.py
     ```
   - Test the FastAPI endpoint:
     ```bash
     uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8080
     ```

7. **Deploy to Cloud Run** (optional):
   - Ensure `gcloud` is installed and authenticated.
   - Run:
     ```bash
     cd rag-agent
     bash deploy.sh
     ```

## Troubleshooting
- **Pandoc/XeLaTeX Errors**:
  - If fonts are missing, ensure `texlive-fonts-extra` is installed.
  - For `undefined control sequence` errors, run `xelatex` twice to resolve references:
    ```bash
    xelatex ai-agent-bootcamp.tex
    xelatex ai-agent-bootcamp.tex
    ```
- **Pinecone Dimension Mismatch**:
  - Verify the embedding model dimension (1536 for `text-embedding-3-small`) matches the Pinecone index.
- **Import Errors**:
  - Ensure all dependencies in `requirements.txt` are installed.
  - Check Python version (3.10+ recommended).
- **Cloud Run Deployment**:
  - Use GCP Secret Manager for API keys instead of hardcoding in `.env`.
  - Increase memory if embeddings fail (`--memory=1Gi` in `deploy.sh`).

## Notes
- The `ai-agent-bootcamp.tex` file includes the full response with proper headings, formatted for PDF output using XeLaTeX and the Noto Serif font.
- The `data/` folder is empty; add your own PDFs/TXTs/CSVs for testing.
- The repository includes small improvements (e.g., consistent error handling in `ingest.py`, clear README).
- If you encounter issues, paste error logs, and I’ll help debug.
- To proceed to the next lesson (multi-agent orchestration with BigQuery), let me know!