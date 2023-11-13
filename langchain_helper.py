import os
import google.generativeai as palm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

from langchain.llms import GooglePalm
llm = GooglePalm(GOOGLE_API_KEY = os.getenv("AIzaSyD_oyQc6pPZeaH8oZOosvaev2IMn7BKF2"), temperature=0.9, max_tokens=500)

loader = CSVLoader(file_path="data.csv", source_column="prompt")
docs = loader.load()

face_embeddings = HuggingFaceEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="data.csv", source_column="prompt")
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=docs,embedding= face_embeddings)
    vectordb.save_local(vectordb_file_path)
    

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, face_embeddings)
    
    # create a retriever for querying the vector database 
    retriever = vectordb.as_retriever(score_threshold=0.7)
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))
