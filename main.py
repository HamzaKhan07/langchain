from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import streamlit as st
import gc

st.header("Talk with PDF")


def load_result():
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.load_local('faiss_index', embeddings)

    # load garbage collector
    gc.enable()

    # question answer chain
    llm = HuggingFaceHub(repo_id="google/flan-ul2", huggingfacehub_api_token="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")
    chain = load_qa_chain(llm=llm, chain_type="stuff")

    similar_chunks = vectordb.similarity_search(query=query, k=2)
    response = chain.run(input_documents=similar_chunks, question=query)

    st.write(response)
    # references
    st.write(similar_chunks[0])
    st.write(similar_chunks[1])

    # delete all the unused vars
    del llm
    del chain
    del similar_chunks
    del response

    # release garbage collector
    gc.collect()


if __name__ == '__main__':

    query = st.text_input("Enter your question")
    if query:
        load_result()

