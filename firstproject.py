from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from getpass import getpass
import os
from langchain_ai21 import AI21SemanticTextSplitter

# llm = Ollama(model="llama2")

llm = ChatOpenAI(openai_api_key="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")

text_splitter = RecursiveCharacterTextSplitter() # Note to Adi: Experiment with some other levels of splitting, Semantic Chunking seems cool

semantic_text_splitter = AI21SemanticTextSplitter(api_key = "zfI6SlDN1ztRepDolMgeMejbOgNUR7xD")

output_parser = StrOutputParser()

embeddings = OpenAIEmbeddings(openai_api_key="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")

# RAG STEPSS:

# Document Loader (Simplest, just getting text from a source): Currently using Web Base Loader, other possible are TextLoader, PDF Loader, or 3rd party loaders such as Amazon Textract Loader

loader = WebBaseLoader("https://www.whitehouse.gov/about-the-white-house/presidents/george-washington/")

docs = loader.load()

# Transformers (Such as google translate, HTML to text, and Doctran converting to Q&A format):

# Text Splitter (Self-proclaimed coolest step) : 

documents = text_splitter.split_documents(docs)

chunks = semantic_text_splitter.split_documents(docs)

# Embedding: Converting text to vectors

# Vector Storage, adding the vectors to a FAISS Storage:

vector = FAISS.from_documents(documents, embeddings) 

# Retreiver: Retrieving the vectors from the storage and filtering info based off the retriever:

retriever = vector.as_retriever()

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class historia n."),
#     ("user", "{input}")
# ])

# chain = prompt | llm | output_parser  # Basic Chain: No files used. Prompt (including message history) + llm call + string parser


prompt = ChatPromptTemplate.from_template("""Answer the following question using this document:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


response = retrieval_chain.invoke({"input": "Using this, summarize George Washington's life"})
print(response["answer"])



