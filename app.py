import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

st.set_page_config(page_title="Proton - Asistente Universitario", layout="wide")
st.title("🤖 Proton - Asistente de la carrera de Ingeniería Eléctrica")
st.markdown("Haz preguntas sobre el decreto oficial de la carrera. Por ejemplo: *¿Cuál es la duración del plan de estudios?*")

# Cargar y procesar el PDF
with st.spinner("Procesando el decreto..."):
    loader = loader = PDFlumberLoader ("14-2025_Decreto.pdf")
    pages = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Preguntas
query = st.text_input("Escribe tu pregunta aquí:")
if query:
    with st.spinner("Buscando respuesta..."):
        result = qa.run(query)
        st.success(result)
