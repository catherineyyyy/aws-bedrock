import json
import os
import sys
import boto3
import streamlit as st

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models.bedrock import BedrockChat

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
# from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from pypdf import PdfReader

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


#Extract PDF Data
def extract_pdf(filename):
    reader = PdfReader(filename)
    page = reader.pages[0]
    return page.extract_text()

## Data ingestion
def data_ingestion(inp):
    loader=PyPDFDirectoryLoader(inp)
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs, inp):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local(f"faiss_index_{inp}")

def get_claude_llm():
    ##create the Anthropic Model
    llm=BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,
                model_kwargs={'max_tokens':1000})
    
    return llm

def get_llama2_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
                model_kwargs={'max_gen_len':200})
    
    return llm


prompt_template_chat = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


prompt_template_compliance = """
Imagine you are a compliance officer for a bank checking if policies and guidelines are being met.
Check the sections of the following document on whether the policies are being met.
<question>
{question}
</question

The following are the poilicies to be checked against:
<context>
{context}
</context

Provide the reason for non compliance with the corresponding section of the document 
and suggest edits to be made. Be as granular as possible.

Assistant:"""

PROMPT1 = PromptTemplate(
    template=prompt_template_compliance, input_variables=["context", "question"]
)

PROMPT2 = PromptTemplate(
    template=prompt_template_chat, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query, PROMPT):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Team LLM")

    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    tab1, tab2, tab3 = st.tabs(["Compi-Ease", "Compliance-Bot", "Regu-sinc"])
    with tab1:
        # user_question = st.text_input("Ask a Question from the PDF Files")
        
        if uploaded_file is not None:
            user_question = extract_pdf(uploaded_file)

        with st.sidebar:
            st.title("Update Or Create Vector Store:")
            
            if st.button("Vectors Update Guidelines"):
                with st.spinner("Processing..."):
                    docs = data_ingestion('guidelines')
                    get_vector_store(docs, 'guidelines')
                    st.success("Done")

        if st.button("Claude Output"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index_guidelines", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm=get_claude_llm()
                
                st.write(get_response_llm(llm,faiss_index,user_question, PROMPT1))
                st.success("Done")

    with tab2:
        st.header("Compliance-Bot")


    with tab3:
        st.header("Regu-sinc")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

    


if __name__ == "__main__":
    main()














