import streamlit as st
import gc
import os, glob
from dotenv import load_dotenv

st.header("Talk with PDF")
load_dotenv()


def load_result(query):
    from langchain.embeddings import GooglePalmEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.llms import GooglePalm

    # load garbage collector
    gc.enable()

    # move imports inside the function

    embeddings = GooglePalmEmbeddings()
    vectordb = FAISS.load_local('palm_index', embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # question answer chain
    llm = GooglePalm()
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    response = chain(query)

    st.write(response)

    # put references here

    # delete all the unused vars
    del embeddings
    del vectordb
    del llm
    del chain
    del response

    # release garbage collector
    gc.collect()


if __name__ == '__main__':
    query = st.text_input("Enter your question")
    if query:
        load_result(query)
