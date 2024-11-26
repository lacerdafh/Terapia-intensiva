import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import SecretStr
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pathlib import Path
import tomllib
import warnings
import pickle
import shutil

# Suprimir avisos
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carregar configurações
with open("config.toml", "rb") as f:
    config = tomllib.load(f)

os.environ["GROQ_API_KEY"] = config["api_keys"]["groq_api_key"]
os.environ["HF_API_KEY"] = config["api_keys"]["hf_api_key"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
load_dotenv()

# Verificar a chave API
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY não encontrada no arquivo .env")

# Inicializar o modelo Groq
chat_model = ChatGroq(
    model_name="llama-3.2-3b-preview",
    temperature=0.7,
    max_tokens=512
)

@st.cache_resource
def get_embeddings():
    """Inicializa e retorna o modelo de embeddings usando HuggingFace Inference API."""
    try:
        hf_api_key = os.getenv("HF_API_KEY")
        if not hf_api_key:
            raise ValueError("A chave da API do Hugging Face (HF_API_KEY) não foi encontrada no .env")

        return HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_api_key,  # Removido SecretStr
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    except Exception as e:
        st.error(f"Erro ao carregar embeddings da HuggingFace API: {str(e)}")
        raise

def get_user_directories() -> tuple[str, str, str]:
    """Configura os diretórios da aplicação na pasta do usuário."""
    if os.name == 'nt':  # Windows
        base_dir = os.path.expanduser("~\\Documents\\ChatbotDrKinho")
    else:  # Linux/Mac
        base_dir = os.path.expanduser("~/ChatbotDrKinho")
    
    docs_dir = os.path.join(base_dir, "documentos")
    index_dir = os.path.join(base_dir, "vector_store")
    
    # Criar diretórios se não existirem
    for directory in [base_dir, docs_dir, index_dir]:
        os.makedirs(directory, exist_ok=True)
        
    return base_dir, docs_dir, index_dir

def load_documents(folder_path: str) -> list[Document]:
    """Carrega documentos TXT e PDF de uma pasta."""
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if file_name.lower().endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path}
                    ))
            elif file_name.lower().endswith(".pdf"):
                pdf_reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() for page in pdf_reader.pages)
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_path}
                ))
        except Exception as e:
            st.warning(f"Erro ao carregar {file_name}: {str(e)}")
    return documents

@st.cache_resource
def create_or_load_vector_store(_embeddings, docs_dir: str, index_dir: str):
    """Cria ou carrega o índice FAISS no diretório local."""
    index_path = os.path.join(index_dir, "faiss_index")
    pickle_path = os.path.join(index_dir, "faiss_store.pkl")

    try:
        # Tentar carregar índice existente
        if os.path.exists(index_path) and os.path.exists(pickle_path):
            st.info("Carregando índice FAISS existente...")
            vector_store = FAISS.load_local(index_path, _embeddings)
            with open(pickle_path, 'rb') as f:
                vector_store.__dict__.update(pickle.load(f))
            return vector_store

        # Criar novo índice
        st.info("Criando novo índice FAISS...")
        documents = load_documents(docs_dir)
        if not documents:
            raise RuntimeError("Nenhum documento válido encontrado.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Criar vetor store
        vector_store = FAISS.from_documents(texts, _embeddings)
        
        # Salvar índice e metadados
        vector_store.save_local(index_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump(vector_store.__dict__, f)
        
        return vector_store

    except Exception as e:
        raise RuntimeError(f"Erro ao criar/carregar o índice FAISS: {str(e)}")

def upload_files(uploaded_files, docs_dir: str) -> list[str]:
    """Salva múltiplos arquivos enviados no diretório de documentos."""
    saved_files = []
    for uploaded_file in uploaded_files:
        try:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Erro ao salvar arquivo {uploaded_file.name}: {e}")
    return saved_files

def main():
    st.title("Chatbot com Dr. Kinho")

    # Configurar diretórios do usuário
    base_dir, docs_dir, index_dir = get_user_directories()

    # Obter embeddings
    embeddings = get_embeddings()

    with st.sidebar:
        image_path = Path(__file__).parent / "static" / "images" / "app_header.png"
        if image_path.exists():
            st.image(str(image_path), caption="Dr. Kinho", use_container_width=True)
        
        st.header("Gerenciamento de Documentos")

        # Upload de documentos
        uploaded_files = st.file_uploader(
            "Envie documentos (TXT ou PDF)",
            type=["txt", "pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            with st.spinner("Salvando arquivos..."):
                saved_files = upload_files(uploaded_files, docs_dir)
                if saved_files:
                    st.success(f"Arquivos salvos: {', '.join(saved_files)}")
                    st.cache_resource.clear()
                    st.rerun()

        # Listar documentos existentes
        st.header("Documentos Disponíveis")
        files = os.listdir(docs_dir)

        if files:
            file_data = [
                {
                    "Arquivo": file,
                    "Tamanho (KB)": f"{os.path.getsize(os.path.join(docs_dir, file)) / 1024:.1f}"
                }
                for file in files
            ]

            st.dataframe(
                file_data,
                hide_index=True,
                use_container_width=True
            )

            # Opção para deletar arquivos
            files_to_delete = st.multiselect("Selecione arquivos para deletar:", files)

            if files_to_delete and st.button("Deletar Selecionados"):
                with st.spinner("Deletando arquivos..."):
                    for file in files_to_delete:
                        try:
                            os.remove(os.path.join(docs_dir, file))
                            st.success(f"Arquivo deletado: {file}")
                        except Exception as e:
                            st.error(f"Erro ao deletar {file}: {e}")
                    st.cache_resource.clear()
                    st.rerun()

        else:
            st.info("Nenhum documento carregado.")

        # Recriar índice vetorial
        if st.button("Recriar Banco de Dados"):
            with st.spinner("Recriando índice vetorial..."):
                try:
                    # Limpar diretório do índice
                    shutil.rmtree(index_dir, ignore_errors=True)
                    os.makedirs(index_dir, exist_ok=True)
                    
                    st.cache_resource.clear()
                    st.session_state.vector_store = create_or_load_vector_store(
                        _embeddings=embeddings,
                        docs_dir=docs_dir,
                        index_dir=index_dir
                    )
                    st.success("Banco de dados recriado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao recriar banco de dados: {e}")
                st.rerun()

        # Mostrar informações dos diretórios
        st.header("Informações")
        st.write(f"📁 Base: {base_dir}")
        st.write(f"📄 Documentos: {docs_dir}")
        st.write(f"📊 Índices: {index_dir}")

    # Configurar banco de dados (FAISS)
    try:
        if 'vector_store' not in st.session_state:
            with st.spinner("Configurando banco de dados..."):
                st.session_state.vector_store = create_or_load_vector_store(
                    _embeddings=embeddings,
                    docs_dir=docs_dir,
                    index_dir=index_dir
                )

        # Configurar o retriever para busca
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})

        # Interface de chat
        user_question = st.text_input("Faça sua pergunta sobre os documentos:")
        
        if user_question:
            with st.spinner("Processando..."):
                context = retriever.get_relevant_documents(user_question)

                chat_model = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model_name="llama-3.2-3b-preview",
                    temperature=0.4,
                    max_tokens=512
                )

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

    except Exception as e:
        st.error(f"Erro: {str(e)}")

if __name__ == "__main__":
    main()
