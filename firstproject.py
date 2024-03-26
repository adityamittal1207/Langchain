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


# llm = Ollama(model="llama2")

llm = ChatOpenAI()

text_splitter = RecursiveCharacterTextSplitter() # Note to Adi: Experiment with some other levels of splitting, Semantic Chunking seems cool

output_parser = StrOutputParser()

embeddings = OpenAIEmbeddings()

loader = WebBaseLoader("<Any link>")

docs = loader.load()

documents = text_splitter.split_documents(docs)

# Note:Try other retrievers, Multi-Query Retriever seems like the best technical retriever

vector = FAISS.from_documents(documents, embeddings) 

retriever = vector.as_retriever()

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are world class historian."),
#     ("user", "{input}")
# ])

# chain = prompt | llm | output_parser  # Basic Chain: No files used. Prompt (including message history) + llm call + string parser


prompt = ChatPromptTemplate.from_template("""Answer the following question based off your knowledge supplemented by this context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


response = retrieval_chain.invoke({"input": "<Question>"})
print(response["answer"])



