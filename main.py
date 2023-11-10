from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import streamlit as st
from PyPDF2 import PdfReader
import gc

st.header("Talk with PDF")

pdf = st.file_uploader('Upload your PDF!')


def load_pdf():
    # load garbage collector
    gc.enable()

    text = ''
    if pdf is not None:
        st.write(pdf)
        pdf_object = PdfReader(pdf)
        for page in pdf_object.pages[:50]:
            text = text + page.extractText()

        # divide text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)
        print(len(chunks))

        # create embeddings
        embeddings = HuggingFaceEmbeddings()

        # store embeddings
        db = FAISS.from_texts(chunks, embedding=embeddings)

        # delete all the unused vars
        del text
        del text_splitter
        del chunks
        del embeddings
        del pdf_object

        # release garbage collector
        gc.collect()

        return db


def load_result(stored_db):
    # load garbage collector
    gc.enable()

    # question answer chain
    llm = HuggingFaceHub(repo_id="google/flan-ul2", huggingfacehub_api_token="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")
    chain = load_qa_chain(llm=llm, chain_type="stuff")

    similar_chunks = stored_db.similarity_search(query=query, k=2)
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
    vector_store = load_pdf()

    query = st.text_input("Enter your question")
    if query:
        load_result(vector_store)

