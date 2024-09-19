import os
from typing import List
from PyPDF2 import PdfReader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.schema import StrOutputParser
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)

import chainlit as cl


class PDFFileLoader:
    def __init__(self, path: str):
        self.documents = []
        self.path = path

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

    def load_file(self):
        with open(self.path, "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            self.documents.append(text)

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        pdf_reader = PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        self.documents.append(text)

    def load_documents(self):
        self.load()
        return self.documents


pdf_loader_NIST = PDFFileLoader("data/NIST.AI.600-1.pdf")
pdf_loader_Blueprint = PDFFileLoader("data/Blueprint-for-an-AI-Bill-of-Rights.pdf")
documents_NIST = pdf_loader_NIST.load_documents()
documents_Blueprint = pdf_loader_Blueprint.load_documents()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents_NIST = text_splitter.split_text(documents_NIST)
split_documents_Blueprint = text_splitter.split_text(documents_Blueprint)
documents = split_documents_NIST + split_documents_Blueprint

embeddings = OpenAIEmbeddings()
# Create a metadata for each chunk
metadatas = [{"source": f"{i}-pl"} for i in range(len(documents))]

# Set up prompts
RAG_PROMPT_TEMPLATE = """ \
Use the provided context to answer the user's query.

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".
"""

USER_PROMPT_TEMPLATE = """ \
Context:
{context}
User Query:
{user_query}
"""

rag_prompt = SystemMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
user_prompt = HumanMessagePromptTemplate.from_template(USER_PROMPT_TEMPLATE)
chat_prompt = ChatPromptTemplate.from_messages([rag_prompt, user_prompt])

@cl.on_chat_start
async def start_chat():
    # settings = {
    #     "model": "gpt-4o-mini",
    #     "temperature": 0,
    #     "max_tokens": 500,
    #     "top_p": 1,
    #     "frequency_penalty": 0,
    #     "presence_penalty": 0,
    # }

    # cl.user_session.set("settings", settings)


    # Vector Database
    docsearch = await cl.make_async(Chroma.from_texts)(
        documents, embeddings, metadatas=metadatas
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4o-mini", streaming=True),
        prompt=chat_prompt, 
        output_parser=StrOutputParser(),
        retriever=docsearch.as_retriever()
    )

    # chain = LLMChain(llm= ChatOpenAI(model_name="gpt-4o-mini", streaming=True), prompt=chat_prompt, output_parser=StrOutputParser())

    cl.user_session.set("chain", chain)

@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content)

    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()