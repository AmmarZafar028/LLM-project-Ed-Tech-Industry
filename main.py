import streamlit as st
import google.generativeai as palm
from langchain_helper import create_vector_db, get_qa_chain

st.title("LangChain Q&A 🌱")
btn = st.button("Create Vector DB")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])  


