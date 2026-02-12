from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

# Mount static files (use paths relative to this file)
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

load_dotenv(BASE_DIR / ".env")

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"Add it to {BASE_DIR / '.env'} or your shell environment."
        )
    return value


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _build_rag_chain() -> Any:
    # Ensure env vars are present (and set for downstream libraries)
    os.environ["PINECONE_API_KEY"] = _require_env("PINECONE_API_KEY")
    os.environ["OPENAI_API_KEY"] = _require_env("OPENAI_API_KEY")

    embeddings = download_hugging_face_embeddings()

    index_name = "medical-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    chat_model = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # LCEL RAG chain: retriever -> format docs -> prompt -> llm -> string
    return (
        {"context": retriever | _format_docs, "input": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )


@app.on_event("startup")
def _startup() -> None:
    # Build heavy objects once at startup (avoids import-time failures)
    app.state.rag_chain = _build_rag_chain()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get", response_class=JSONResponse)
async def chat(msg: str = Form(...)):
    rag_chain = getattr(app.state, "rag_chain", None)
    if rag_chain is None:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG chain not initialized. Check server logs for startup errors."},
        )

    print(msg)
    response = rag_chain.invoke(msg)
    print("Response : ", response)
    return JSONResponse(content={"answer": str(response)})

# To run: uvicorn app:app --host 0.0.0.0 --port 8080 --reload   