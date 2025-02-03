import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
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

# Inicializar modelo de chat Groq
chat_model = ChatGroq(
    model_name="llama-3.2-3b-preview",
    temperature=0.3,
    max_tokens=512
)

@st.cache_resource
def get_embeddings():
    """Inicializa e retorna o modelo de embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_app_directories() -> tuple[str, str, str]:
    """Configura diretórios para documentos e índices."""
    base_dir = os.path.expanduser(r"D:\6-projetos\8-ChatbotUTI\Terapia-intensiva\assets")
    docs_dir = os.path.join(base_dir)
    index_dir = os.path.join(base_dir, "vector_store")
    for directory in [base_dir, docs_dir, index_dir]:
        os.makedirs(directory, exist_ok=True)
    return base_dir, docs_dir, index_dir

def load_documents(folder_path: str) -> list[Document]:
    """Carrega documentos TXT e PDF."""
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if file_name.lower().endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    documents.append(Document(page_content=text, metadata={"source": file_path}))
            elif file_name.lower().endswith(".pdf"):
                pdf_reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                documents.append(Document(page_content=text, metadata={"source": file_path}))
        except Exception as e:
            st.warning(f"Erro ao carregar {file_name}: {str(e)}")
    return documents

@st.cache_resource
def create_or_load_vector_store(_embeddings, docs_dir: str, index_dir: str, force_recreate=False):
    """Cria ou carrega o índice FAISS."""
    index_file = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_file) and not force_recreate:
        st.info("Carregando índice FAISS existente...")
        return FAISS.load_local(index_dir, _embeddings, allow_dangerous_deserialization=True)
    st.info("Criando novo índice FAISS...")
    documents = load_documents(docs_dir)
    if not documents:
        raise RuntimeError("Nenhum documento válido encontrado.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, _embeddings)
    vector_store.save_local(index_dir)
    return vector_store

def upload_files(uploaded_files, docs_dir: str) -> list[str]:
    """Salva arquivos enviados no diretório."""
    saved_files = []
    st.write(f"Diretório de destino: {docs_dir}")
    for uploaded_file in uploaded_files:
        try:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            st.write(f"Salvando arquivo em: {file_path}")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Erro ao salvar arquivo {uploaded_file.name}: {e}")
    return saved_files

def main():
    st.title("Chatbot do internato em UTI")
    base_dir, docs_dir, index_dir = get_app_directories()
    embeddings = get_embeddings()

    with st.sidebar:
        st.image("D:/6-projetos/8-ChatbotUTI/Terapia-intensiva/static/images/app_header.png", use_container_width=True)
        st.header("Trabalhos em Discussão")
        st.text("""
        1- Daily laxative therapy reduces organ dysfunction in mechanically ventilated patients.
        2- Evaluation of simplified acute physiology score 3 performance.
        """)
        st.markdown("""
        **Como funciona este chatbot?**
        - Os documentos carregados são processados e indexados para busca eficiente.  
        - Quando uma pergunta é feita, o chatbot busca informações relevantes nos documentos.  
        - Ele usa inteligência artificial para encontrar os melhores trechos como base de resposta.  
        """)

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = create_or_load_vector_store(embeddings, docs_dir, index_dir)
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})

    user_question = st.text_input("Faça sua pergunta:")
    if user_question:
        context = retriever.get_relevant_documents(user_question)
        messages = [
            ("system", "Você é um assistente que responde com base no contexto fornecido."),
            ("user", f"Contexto: {' '.join(doc.page_content for doc in context)}\nPergunta: {user_question}")
        ]
        response = chat_model.invoke(messages)
        st.markdown("### Resposta:")
        st.write(response.content)

if __name__ == "__main__":
    main()
