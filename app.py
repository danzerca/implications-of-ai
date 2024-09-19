# from typing import List
# from chainlit.types import AskFileResponse
from aimakerspace.text_utils import CharacterTextSplitter, PDFFileLoader
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
)
# from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.vectordatabase import VectorDatabase
# from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import chainlit as cl
# import asyncio
import nest_asyncio
nest_asyncio.apply()
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


filepath_NIST = "data/NIST.AI.600-1.pdf"
filepath_Blueprint = "data/Blueprint-for-an-AI-Bill-of-Rights.pdf"

documents_NIST = PyMuPDFLoader(filepath_NIST).load()
documents_Blueprint = PyMuPDFLoader(filepath_Blueprint).load()
documents = documents_NIST + documents_Blueprint


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

rag_documents = text_splitter.split_documents(documents)

RAG_PROMPT_TEMPLATE = """ \
Use the provided context to answer the user's query.
You may not answer the user's query unless there is specific context in the following text.
If you do not know the answer, or cannot answer, please respond with "I don't know".
"""

rag_prompt = SystemRolePrompt(RAG_PROMPT_TEMPLATE)

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
    vector_db = await vector_db.abuild_from_list(rag_documents)
    # vector_db = await vector_db.abuild_from_list(split_documents_NIST)
    # vector_db = await vector_db.abuild_from_list(split_documents_Blueprint)
    
    # chat_openai = ChatOpenAI()
    llm = ChatOpenAI(model="gpt-4o-mini", tags=["base_llm"]) 


    # Create a chain
    retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=llm
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