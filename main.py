import streamlit as st
import gc

st.header("Talk with PDF")


def load_result(query):
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.llms import HuggingFaceHub

    # load garbage collector
    gc.enable()

    # move imports inside the function

    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.load_local('faiss_index', embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # question answer chain
    llm = HuggingFaceHub(repo_id="google/flan-ul2", huggingfacehub_api_token="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")
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
