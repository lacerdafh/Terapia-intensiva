import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from groq import Groq

# Carregar variáveis de ambiente
load_dotenv()

# Configurações
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")

# Criar diretórios se não existirem
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# Inicialização do estado da sessão
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def load_documents(folder_path: str) -> list[Document]:
    """Carrega documentos TXT e PDF de uma pasta."""
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if file_name.lower().endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file_name.lower().endswith(".pdf"):
                pdf_reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() for page in pdf_reader.pages)
                documents.append(Document(page_content=text, metadata={"source": file_path}))
        except Exception as e:
            st.warning(f"Erro ao carregar {file_name}: {str(e)}")
    return documents

@st.cache_resource
def create_or_load_vector_store():
    """Cria ou carrega o índice FAISS com cache."""
    index_path = os.path.join(FAISS_INDEX_PATH, "index")
    try:
        if os.path.exists(index_path):
            st.info("Carregando índice FAISS existente...")
            return FAISS.load_local(
                index_path,
                OllamaEmbeddings(model="llama3.2:3b"),
                allow_dangerous_deserialization=True
            )
        
        st.info("Criando novo índice FAISS...")
        documents = load_documents(DOCUMENTS_FOLDER)
        if not documents:
            raise RuntimeError("Nenhum documento válido encontrado.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(
            texts,
            OllamaEmbeddings(model="llama3.2:3b")
        )
        
        # Salvar o índice
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Erro ao criar/carregar o vector store: {str(e)}")

def main():
    st.title("Chatbot baseado em documentos locais")
    
    # Sidebar para controles
    with st.sidebar:
        st.header("Controles")
        if st.button("Recriar Índice"):
            # Limpar o cache e forçar recriação do índice
            st.cache_resource.clear()
            st.session_state.vector_store = None
            st.experimental_rerun()
            
        if st.button("Ver Documentos"):
            files = os.listdir(DOCUMENTS_FOLDER)
            if files:
                st.write("Documentos disponíveis:")
                for file in files:
                    st.write(f"- {file}")
            else:
                st.warning("Nenhum documento encontrado")

    try:
        # Carregar ou criar vector store
        if not st.session_state.vector_store:
            with st.spinner("Configurando banco de dados..."):
                st.session_state.vector_store = create_or_load_vector_store()
                st.success("Banco de dados configurado!")

        # Configurar retriever
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )

        # Configurar modelo e template
        template = ChatPromptTemplate.from_template("""
        Responda a pergunta baseando-se apenas no seguinte contexto:
        {context}

        Pergunta: {question}
        """)
        
        model = Groq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.2-3b-preview",
            temperature=0.4,
            max_tokens=1024
        )

        # Interface principal
        user_question = st.text_input("Querido Kinho, Faça sua pergunta:")
        if user_question:
            with st.spinner("Processando..."):
                context = retriever.get_relevant_documents(user_question)
                prompt = template.format(
                    context="\n".join(doc.page_content for doc in context),
                    question=user_question
                )
                response = model.invoke(prompt)
                st.write("Resposta:", response)

    except Exception as e:
        st.error(f"Erro: {str(e)}")

if __name__ == "__main__":
    main()