from langchain_community.document_loaders import PyPDFLoader, TextLoader,SeleniumURLLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st
import langsmith
import pathlib
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain_perplexity import ChatPerplexity
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv()
import io
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain # 
from langchain.chains import create_retrieval_chain # combine retriever and LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda,RunnableParallel
import uuid
os.environ['perplexity_api_key'] = os.getenv('PERPLEXITY_API_KEY')

os.environ["LANGCHAIN_PROJECT"] = 'pdf_chat_app'


st.set_page_config(
    page_title="PDF Chat Bot",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('PDF Chat Bot with Perplexity AI')

api_key=st.text_input("Enter your Perplexity API Key:", type="password",help="You can get your API key from https://www.perplexity.ai/")
openai_key=st.text_input("Enter your OpenAI API Key:", type="password",help="You can get your API key from https://platform.openai.com/account/api-keys")
if openai_key:
    os.environ['OPENAI_API_KEY']=openai_key
if not api_key:
    api_key=os.getenv("PERPLEXITY_API_KEY")

temp=st.slider("Select Temperature:", min_value=0.0, max_value=2.0, step=0.1, value=0.7)

# upload PDF file  or url 
option=st.selectbox("Select the type of document you want to upload:", options=["PDF", "Webpage"])

# define LLM

llm = ChatPerplexity(
    model="sonar",
    temperature=temp,
    pplx_api_key=os.getenv("PERPLEXITY_API_KEY") if api_key=="" else api_key)

# Initialize session state for storing vectors when we keel it global and create FAISS only once
if 'url_vectors' not in st.session_state:
    st.session_state['url_vectors'] = None
if 'pdf_vectors' not in st.session_state:  
    st.session_state['pdf_vectors'] = None


if option=="PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])  # uploaded file is a BytesIO object. It means it stores the file in memory

    if uploaded_file is not None:
     with st.spinner("Processing PDF ..."):  
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)   #  It expects a file path, not a BytesIO object
        loaders = loader.load()
        st.success("PDF loaded successfully!")
        # Display the content of the PDF
        st.write("PDF Content:")
        
        splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        chunks=splitter.split_documents(loaders)  # split_documents is used to split the document into smaller chunks
        embeddings=OpenAIEmbeddings(api_key=openai_key if openai_key else os.getenv("OPENAI_API_KEY"))
        st.session_state.pdf_vectors=FAISS.from_documents(chunks, embeddings)
        st.session_state.pdf_vectors.save_local("pdf_vectors")

        st.success("Embeddings generated and saved successfully!")
    
    st.subheader("Ask questions about the PDF")
    user_question=st.text_input("Enter your question:")
    if st.button("Get Answer") and user_question:
        if st.session_state.pdf_vectors:
            vectors = st.session_state.pdf_vectors
            retriever = vectors.as_retriever(search_kwargs={"k": 3})

            def format_docs(docs):
                return "\n".join([doc.page_content for doc in docs])

            parallel = RunnableParallel(
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }
            )
            # Create a prompt template
            prompt=ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant that answers questions based on the provided context."),
                    ("human", "context: {context}\n\nquestion: {question}"),
                ])
            chain=parallel | prompt | llm | StrOutputParser()
            result=chain.invoke(user_question)
            with st.spinner("Getting Answer..."):
                st.subheader("Answer:")
                st.write(result)
                

else:  # Webpage case
    url = st.text_input("Enter URL of the Webpage:")

    if url:
        with st.spinner("Fetching and Processing Webpage..."):
            loader = SeleniumURLLoader(urls=[url])
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            st.session_state.url_vectors = FAISS.from_documents(chunks, embeddings)
            st.session_state.url_vectors.save_local("url_vectors")
            st.success("Webpage content processed and embeddings saved!")

    user_question = st.text_input("Enter your question about the webpage:")

    

    if st.button("Get Answer (Webpage)") and user_question and st.session_state.url_vectors:
        vectors = st.session_state.url_vectors
        retriever = vectors.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs):
            return "\n".join([doc.page_content for doc in docs])

        parallel = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context."),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])

        chain = parallel | prompt | llm | StrOutputParser()
        with st.spinner("Getting Answer..."):
            result = chain.invoke(user_question)
        st.subheader("Answer:")
        st.write(result)


