import os
from typing import List
from chainlit.types import AskFileResponse
from aimakerspace.text_utils import CharacterTextSplitter, PDFFileLoader
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import chainlit as cl
import asyncio
import nest_asyncio
nest_asyncio.apply()
import langchain_community
from langchain_community.document_loaders import PyMuPDFLoader
import langchain
from langchain.prompts import ChatPromptTemplate

filepath_NIST = "data/NIST.AI.600-1.pdf"
filepath_Blueprint = "data/Blueprint-for-an-AI-Bill-of-Rights.pdf"

documents_NIST = PyMuPDFLoader(filepath_NIST).load()
documents_Blueprint = PyMuPDFLoader(filepath_Blueprint).load()
documents = documents_NIST + documents_Blueprint

# pdf_loader_NIST = PDFFileLoader("data/NIST.AI.600-1.pdf")
# pdf_loader_Blueprint = PDFFileLoader("data/Blueprint-for-an-AI-Bill-of-Rights.pdf")
# documents_NIST = pdf_loader_NIST.load_documents()
# documents_Blueprint = pdf_loader_Blueprint.load_documents()

# text_splitter = CharacterTextSplitter()
# split_documents_NIST = text_splitter.split_texts(documents_NIST)
# split_documents_Blueprint = text_splitter.split_texts(documents_Blueprint)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

rag_documents = text_splitter.split_documents(documents)

RAG_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

USER_PROMPT_TEMPLATE = """ \
Context:
{context}

User Query:
{user_query}
"""

user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    async def arun_pipeline(self, user_query: str):
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)

        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = rag_prompt.create_message()

        formatted_user_prompt = user_prompt.create_message(user_query=user_query, context=context_prompt)

        async def generate_response():
            async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
                yield chunk

        return {"response": generate_response(), "context": context_list}


# ------------------------------------------------------------


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    # settings = {
    #     "model": "gpt-3.5-turbo",
    #     "temperature": 0,
    #     "max_tokens": 500,
    #     "top_p": 1,
    #     "frequency_penalty": 0,
    #     "presence_penalty": 0,
    # }

    # Create a dict vector store
    vector_db = VectorDatabase()
    vector_db = await vector_db.abuild_from_list(split_documents_NIST)
    vector_db = await vector_db.abuild_from_list(split_documents_Blueprint)
    
    chat_openai = ChatOpenAI()

    # Create a chain
    retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai
    )

    # cl.user_session.set("settings", settings)
    cl.user_session.set("chain", retrieval_augmented_qa_pipeline)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content)

    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()
