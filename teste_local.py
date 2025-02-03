import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pathlib import Path
import tomllib
import warnings

# Suprimir avisos
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carregar configurações do arquivo config.toml
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

os.environ["GROQ_API_KEY"] = config["api_keys"]["groq_api_key"]
os.environ["HF_API_KEY"] = config["api_keys"]["hf_api_key"]

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY não encontrada no arquivo .env")

# Inicializar modelo de chat Groq
chat_model = ChatGroq(
    model_name="llama-3.2-3b-preview",
    temperature=0.3,
    max_tokens=512
)

@st.cache_resource
def get_embeddings():
    """Inicializa e retorna o modelo de embeddings."""
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        raise ValueError("HF_API_KEY não encontrada no arquivo .env")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Diretório fixo para os documentos
BASE_DIR = "./assets"
DOCS_DIR = BASE_DIR
INDEX_DIR = os.path.join(BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index")

# Criar diretórios se não existirem
os.makedirs(INDEX_DIR, exist_ok=True)

@st.cache_resource
def load_vector_store(_embeddings):
    """Carrega o índice FAISS fixo."""
    if os.path.exists(INDEX_PATH):
        st.success("Índice FAISS carregado com sucesso!")
        return FAISS.load_local(INDEX_PATH, _embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("Índice FAISS não encontrado. Certifique-se de que ele foi treinado e salvo corretamente.")
        return None

def main():
    st.title("Chatbot do internato em UTI")
    
    # Inicializar embeddings e carregar índice FAISS
    embeddings = get_embeddings()
    vector_store = load_vector_store(embeddings)
    
    if vector_store is None:
        st.error("Erro ao carregar o banco de dados. O chatbot não pode funcionar sem o índice FAISS.")
        return
    
    # Sidebar: Mostrar arquivos disponíveis
    with st.sidebar:
        st.header("Documentos Carregados")
        files = os.listdir(DOCS_DIR)
        if files:
            for file in files:
                st.write(f"📄 {file}")
        else:
            st.warning("Nenhum documento encontrado")
    
    # Configurar retriever
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    
    # Caixa de entrada de perguntas
    user_question = st.text_input("Faça sua pergunta sobre os documentos:")
    if user_question:
        with st.spinner("Processando..."):
            context = retriever.get_relevant_documents(user_question)
            messages = [
                ("system", "Você é um assistente que responde com base no contexto fornecido."),
                ("user", f"""
                Contexto: {' '.join(doc.page_content for doc in context)}
                
                Pergunta: {user_question}
                """)
            ]
            
            response = chat_model.invoke(messages)
            
            with st.container():
                st.markdown("### Resposta:")
                st.write(response.content)
                
                st.markdown("#### Fontes consultadas:")
                sources = set(doc.metadata.get('source', 'Desconhecido') for doc in context)
                for source in sources:
                    st.write(f"- {os.path.basename(source)}")

if __name__ == "__main__":
    main()
