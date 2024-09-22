import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_core.output_parsers import StrOutputParser

from operator import itemgetter
import chainlit as cl
from sentence_transformers import SentenceTransformer

# Load all the documents in the directory
documents = []
directory = "data/"

for filename in os.listdir(directory):
    if filename.endswith(".pdf"):  # Check if the file is a PDF
        file_path = os.path.join(directory, filename)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)

# Split the documents by character
character_text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=40,
    length_function=len,
    is_separator_regex=False,
)
rag_documents = character_text_splitter.split_documents(documents)

# Split the documents recursively
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=40,
    length_function=len,
    is_separator_regex=False
)
#rag_documents = recursive_text_splitter.split_documents(documents)


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode(text)

# Use the wrapper class for the fine-tuned model
model = SentenceTransformer("danicafisher/dfisher-sentence-transformer-fine-tuned2")
embedding = SentenceTransformerEmbeddings(model)

# Non-fine-tuned model
# embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Create the vector store
vectorstore = Qdrant.from_documents(
    rag_documents,
    embedding,
    location=":memory:",
    collection_name="Implications of AI",
)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4")


@cl.on_chat_start
async def start_chat():
    template = """
        Use the provided context to answer the user's query.
        You may not answer the user's query unless there is specific context in the following text.
        If you do not know the answer, or cannot answer, please respond with "I don't know".
        Question:
        {question}
        Context:
        {context}
        Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    base_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | prompt | llm | StrOutputParser()
    )

    cl.user_session.set("chain", base_chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    result = chain.invoke({"question":message.content})
    msg = cl.Message(content=result)

    await msg.send()