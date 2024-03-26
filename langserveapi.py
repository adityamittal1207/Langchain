#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


app = FastAPI(
    title="Langchain Physics Tutor",
)

llm = ChatOpenAI()

text_splitter = RecursiveCharacterTextSplitter()

embeddings = OpenAIEmbeddings()

topics = {"Optics": "https://library.fiveable.me/ap-physics-2/unit-6/review/study-guide/uV8klagEJVQYCXfE0a9Z", "Kinematics": "https://courses.lumenlearning.com/suny-physics/chapter/3-4-projectile-motion/", "Fluids": "https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/Book%3A_University_Physics_I_-_Mechanics_Sound_Oscillations_and_Waves_(OpenStax)/14%3A_Fluid_Mechanics/14.S%3A_Fluid_Mechanics_(Summary)"}

prompt = ChatPromptTemplate.from_template("""Answer the following question based purely off the physics content of this context:

<context>
{context}
</context>

Question: {topic}""")


for top in topics:

    loader = WebBaseLoader(topics[top])
    docs = loader.load()
    # print(docs)
    documents = text_splitter.split_documents(docs)
    # print(documents)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    print(document_chain)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    retrieval_chain = retrieval_chain # Convert output to string
    print(retrieval_chain)
    add_routes(
        app,
        retrieval_chain, # This works with most chains, but not with retrieval chains. Something about the lambda input of retrieval chains compared to the normal argument input of other chains
# Issue possible has something to do with chain being declared differently, with its own create function. Maybe not a runnable?
        path=(f"/{top}"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)